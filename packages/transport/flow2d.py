"""Laminar incompressible flow on a 2-D geometry the user describes (planar or axisymmetric), solved with FEniCSx.

The fluid domain is a union of axis-aligned rectangles minus obstacles (rectangles, circles; a circle centred on the
axis of an axisymmetric domain is a sphere). Boundary segments are picked with simple geometric selectors and default
to no-slip walls. The steady Navier-Stokes equations are solved with Taylor-Hood P2-P1 elements and Newton (with
continuation in Re when Newton fails); optionally a semi-implicit BDF2 transient (vortex shedding) and a steady passive
scalar (Nusselt / Sherwood on a surface). The fluid is Newtonian or generalised Newtonian (`rheology`: power law,
Carreau, Carreau-Yasuda; steady only), with the apparent viscosity a function of γ̇ = √(2 D:D).

FEniCSx runs as a SEPARATE PROCESS (`packages/transport/fenicsx/flow2d_solver.py`, JSON in/out) with a timeout that
kills the process group, exactly like `reactor_field`. Nothing from dolfinx is imported here. This module validates
the request, chooses reference scales and mesh sizes, and turns the solver output into evidence:

  * newton_converged: SNES converged at the target Re (residual norm reported; continuation path when used)
  * mass_balanced: |Σ in − Σ out| / flow < MASS_TOL
  * mesh_converged: every reported number changes < mesh tolerance between two meshes (size ratio `refine_ratio`);
    the ones that do not are listed in `not_mesh_converged`
  * poiseuille_self_check: on every straight section where the flow is fully developed, the measured dp/ds equals
    the exact Poiseuille value for the measured flow (None when no section is developed)
`verified` is true only when all hold. For a transient request it also needs the run to be complete (periodic / steady
/ fixed final time reached, not cut by `max_wall_s`) and BOTH transient checks done and passed: time step (2·dt, on by
default) and mesh (coarser mesh, `transient.refinement = "both"`). A run cut by the wall-clock budget is returned with
`partial = True` and its numbers under `transient.provisional_not_final`.
"""
from __future__ import annotations

import math
from pathlib import Path

from packages.control.nmpc import parse_expression
from packages.transport import reactor_field as RF

SOLVER = Path(__file__).with_name("fenicsx") / "flow2d_solver.py"
MASS_TOL = 1e-6
MESH_TOL = 0.02
POISEUILLE_TOL = 0.02
MAX_BASE_CELLS = 20000
TYPES = ("inlet", "outlet", "wall", "symmetry", "stream")
SIDES = {"left": ("x", 0), "right": ("x", 2), "bottom": ("y", 1), "top": ("y", 3)}
EXPR_NAMES = {"x", "y", "r", "z", "pi"}
RHEOLOGY_KEYS = {"power_law": {"model", "K", "n", "gamma_reg_1_s"},
                 "carreau": {"model", "eta0", "eta_inf", "lambda_s", "n"},
                 "carreau_yasuda": {"model", "eta0", "eta_inf", "lambda_s", "n", "a", "gamma_reg_1_s"}}

fenicsx_python = RF.fenicsx_python
fenicsx_available = RF.fenicsx_available


def _num(v, what):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{what}: se esperaba un número, llegó {v!r}")
    return float(v)


def _box(b, what):
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        raise ValueError(f"{what}: se esperan 4 números [x0, y0, x1, y1]")
    x0, y0, x1, y1 = (_num(v, what) for v in b)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"{what}: se requiere x1 > x0 e y1 > y0")
    return [x0, y0, x1, y1]


def _line(on, bbox, rects, what):
    """Normalise a line selector to {axis, value, range}."""
    if not isinstance(on, dict):
        raise ValueError(f"{what}: 'on' debe ser un objeto: {{'side': ...}}, {{'x': c, 'y': [lo, hi]}}, {{'y': c, 'x': [lo, hi]}} u {{'obstacle': nombre}}")
    if "side" in on:
        if on["side"] not in SIDES:
            raise ValueError(f"{what}: side debe ser left, right, bottom o top")
        axis, k = SIDES[on["side"]]
        value, rng = bbox[k], on.get("range")
    elif isinstance(on.get("x"), (int, float)) and not isinstance(on.get("y"), (int, float)):
        axis, value, rng = "x", float(on["x"]), on.get("y")
    elif isinstance(on.get("y"), (int, float)) and not isinstance(on.get("x"), (int, float)):
        axis, value, rng = "y", float(on["y"]), on.get("x")
    else:
        raise ValueError(f"{what}: selector no reconocido {on!r}; usa side, o x = constante con y = [lo, hi], o y = constante con x = [lo, hi]")
    if rng is None:
        # extent of the rectangle edges lying on that line (a side of the bounding box may be only partly fluid)
        k = 0 if axis == "x" else 1
        iv = sorted((r[1 - k], r[3 - k]) for r in rects if abs(r[k] - value) < 1e-12 or abs(r[k + 2] - value) < 1e-12)
        if not iv:
            raise ValueError(f"{what}: ningún rectángulo tiene un lado sobre {axis} = {value:g}")
        merged = [list(iv[0])]
        for a, b in iv[1:]:
            if a <= merged[-1][1] + 1e-12:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        if len(merged) > 1:
            raise ValueError(f"{what}: sobre {axis} = {value:g} hay {len(merged)} tramos separados {merged}; da el intervalo "
                             "('range' con side, o el intervalo de la otra coordenada)")
        rng = merged[0]
    if not isinstance(rng, (list, tuple)) or len(rng) != 2 or not rng[1] > rng[0]:
        raise ValueError(f"{what}: el intervalo debe ser [lo, hi] con hi > lo")
    return {"axis": axis, "value": float(value), "range": [float(rng[0]), float(rng[1])]}


def apparent_viscosity(r: dict, shear_rate_1_s: float) -> float:
    """η(γ̇) in Pa·s of a normalised rheology dict (the same expressions the solver assembles, regularisation included)."""
    g, n = float(shear_rate_1_s), r["n"]
    e2 = r.get("gamma_reg_1_s", 0.0) ** 2
    if r["model"] == "power_law":
        return r["K"] * (g * g + e2) ** ((n - 1) / 2)
    if r["model"] == "carreau":
        return r["eta_inf"] + (r["eta0"] - r["eta_inf"]) * (1 + (r["lambda_s"] * g) ** 2) ** ((n - 1) / 2)
    a = r["a"]
    return r["eta_inf"] + (r["eta0"] - r["eta_inf"]) * (1 + r["lambda_s"] ** a * (g * g + e2) ** (a / 2)) ** ((n - 1) / a)


def _rheology(rheology, rate_ref):
    """Validate and normalise the rheology request. `rate_ref` = U_ref / L_ref (1/s) scales the default regularisation."""
    if not isinstance(rheology, dict) or rheology.get("model") not in RHEOLOGY_KEYS:
        raise ValueError(f"rheology.model debe ser uno de {', '.join(RHEOLOGY_KEYS)}")
    model = rheology["model"]
    unknown = set(rheology) - RHEOLOGY_KEYS[model]
    if unknown:
        raise ValueError(f"rheology ({model}): claves no reconocidas {sorted(unknown)}; válidas: {sorted(RHEOLOGY_KEYS[model])}")
    n = _num(rheology.get("n"), "rheology.n")
    if not 0.05 <= n <= 3:
        raise ValueError("rheology.n (índice de flujo) debe estar entre 0.05 y 3")
    r = {"model": model, "n": n}
    if model == "power_law":
        r["K"] = _num(rheology.get("K"), "rheology.K (Pa·s^n)")
        if r["K"] <= 0:
            raise ValueError("rheology.K debe ser positivo (Pa·s^n)")
        r["gamma_reg_1_s"] = _num(rheology.get("gamma_reg_1_s", 1e-3 * rate_ref), "rheology.gamma_reg_1_s")
        if r["gamma_reg_1_s"] <= 0:
            raise ValueError("rheology.gamma_reg_1_s debe ser positivo: la ley de potencia no está definida en γ̇ = 0")
    else:
        r["eta0"] = _num(rheology.get("eta0"), "rheology.eta0 (Pa·s)")
        r["eta_inf"] = _num(rheology.get("eta_inf", 0.0), "rheology.eta_inf (Pa·s)")
        r["lambda_s"] = _num(rheology.get("lambda_s"), "rheology.lambda_s (s)")
        if r["eta0"] <= 0 or r["eta_inf"] < 0 or r["lambda_s"] < 0:
            raise ValueError("rheology: eta0 > 0, eta_inf ≥ 0 y lambda_s ≥ 0")
        if model == "carreau_yasuda":
            r["a"] = _num(rheology.get("a", 2.0), "rheology.a")
            if r["a"] <= 0:
                raise ValueError("rheology.a (exponente de Yasuda) debe ser positivo")
            r["gamma_reg_1_s"] = _num(rheology.get("gamma_reg_1_s", 1e-6 * rate_ref), "rheology.gamma_reg_1_s")
            if r["gamma_reg_1_s"] <= 0:
                raise ValueError("rheology.gamma_reg_1_s debe ser positivo")
        # exactly Newtonian members of the family: the developed-flow self-check then uses Poiseuille with η0
        r["newtonian_equivalent_Pa_s"] = r["eta0"] if (n == 1 or r["lambda_s"] == 0 or r["eta0"] == r["eta_inf"]) else None
    return r


def _prepare(geometry, boundaries, density_kg_m3, viscosity_Pa_s, reynolds, reference_length_m, reference_velocity_m_s,
             stream_direction, points, sections, pressure_drop_between, scalar, transient, mesh, rheology=None):
    if not isinstance(geometry, dict) or geometry.get("mode") not in ("planar", "axisymmetric"):
        raise ValueError("geometry.mode debe ser 'planar' o 'axisymmetric'")
    axi = geometry["mode"] == "axisymmetric"
    rects = [_box(r, f"geometry.rectangles[{i}]") for i, r in enumerate(geometry.get("rectangles") or [])]
    if not rects:
        raise ValueError("geometry.rectangles: falta al menos un rectángulo [x0, y0, x1, y1] de fluido")
    if axi and min(r[1] for r in rects) < 0:
        raise ValueError("axisymmetric: y es el radio (r ≥ 0) y x el eje z; ningún rectángulo puede tener y < 0")
    if stream_direction not in ("+x", "-x", "+y", "-y") or (axi and stream_direction[1] != "x"):
        raise ValueError("stream_direction debe ser +x, -x, +y o -y (en axisymmetric solo ±x, el eje)")
    sax = 0 if stream_direction[1] == "x" else 1
    bbox = [min(r[0] for r in rects), min(r[1] for r in rects), max(r[2] for r in rects), max(r[3] for r in rects)]
    obstacles = []
    for i, o in enumerate(geometry.get("obstacles") or []):
        name = str(o.get("name") or f"obstacle{i + 1}")
        if o.get("type") == "circle":
            c, r = o.get("center"), _num(o.get("radius"), f"obstáculo {name}: radius")
            if not isinstance(c, (list, tuple)) or len(c) != 2 or r <= 0:
                raise ValueError(f"obstáculo {name}: se requiere center [x, y] y radius > 0")
            ob = {"name": name, "type": "circle", "center": [float(c[0]), float(c[1])], "radius": r}
            if axi and 0 < ob["center"][1] < r:
                raise ValueError(f"obstáculo {name}: en axisymmetric el círculo debe estar centrado en el eje (esfera, y = 0) o no tocarlo (toro)")
            size = 2 * r
        elif o.get("type") == "rectangle":
            ob = {"name": name, "type": "rectangle", "box": _box(o.get("box"), f"obstáculo {name}: box")}
            size = min(ob["box"][2] - ob["box"][0], ob["box"][3] - ob["box"][1])
        else:
            raise ValueError(f"obstáculo {name}: type debe ser 'circle' o 'rectangle'")
        ob["_size"] = size
        obstacles.append(ob)
    if len({o["name"] for o in obstacles}) != len(obstacles):
        raise ValueError("los obstáculos deben tener nombres distintos")
    onames = {o["name"] for o in obstacles}

    bnds, used_obs = [], set()
    for i, b in enumerate(boundaries or []):
        name = str(b.get("name") or f"b{i + 1}")
        what = f"frontera '{name}'"
        t = b.get("type")
        if t not in TYPES:
            raise ValueError(f"{what}: type debe ser uno de {', '.join(TYPES)}")
        on = b.get("on") or {}
        nb = {"name": name, "type": t}
        if "obstacle" in on:
            if on["obstacle"] not in onames or t != "wall":
                raise ValueError(f"{what}: on.obstacle debe nombrar un obstáculo existente y la frontera ser 'wall'")
            nb["obstacle"] = on["obstacle"]
            used_obs.add(on["obstacle"])
        else:
            nb.update(_line(on, bbox, rects, what))
        if b.get("velocity") is not None:
            v = b["velocity"]
            if t == "outlet" or t == "symmetry" or not isinstance(v, (list, tuple)) or len(v) != 2:
                raise ValueError(f"{what}: velocity = [vx, vy] (números o expresiones de x, y) solo en inlet, stream o wall")
            for e in v:
                if isinstance(e, str):
                    parse_expression(e, EXPR_NAMES, what)
                else:
                    _num(e, what)
            nb["velocity"] = list(v)
        if t == "inlet":
            if (b.get("mean_velocity") is None) == (b.get("velocity") is None):
                raise ValueError(f"{what}: da mean_velocity (con profile 'parabolic' o 'uniform') o velocity = [vx, vy], no ambos")
            if b.get("mean_velocity") is not None:
                nb["mean_velocity"] = _num(b["mean_velocity"], what)
                nb["profile"] = b.get("profile", "parabolic")
                if nb["profile"] not in ("parabolic", "uniform") or nb["mean_velocity"] <= 0:
                    raise ValueError(f"{what}: profile debe ser 'parabolic' o 'uniform' y mean_velocity > 0 (entra al dominio)")
        if t == "stream" and "velocity" not in nb:
            raise ValueError(f"{what}: una frontera 'stream' necesita velocity = [vx, vy] de la corriente libre")
        if t == "outlet":
            nb["pressure"] = _num(b.get("pressure", 0.0), what)
            nb["normal_flow"] = bool(b.get("normal_flow", True))
        bnds.append(nb)
    for o in obstacles:
        if o["name"] not in used_obs:
            bnds.append({"name": o["name"], "type": "wall", "obstacle": o["name"]})
    names = [b["name"] for b in bnds]
    if len(set(names)) != len(names):
        raise ValueError("las fronteras deben tener nombres distintos (y distintos de los obstáculos)")
    if not any(b["type"] in ("inlet", "stream") or b.get("velocity") is not None for b in bnds):
        raise ValueError("no hay ninguna frontera que mueva el fluido: añade un inlet, un stream o una pared con velocity")

    # reference scales
    U = reference_velocity_m_s
    if U is None:
        cands = [b["mean_velocity"] for b in bnds if b.get("mean_velocity") is not None]
        cands += [math.hypot(*b["velocity"]) for b in bnds if b.get("velocity") is not None
                  and all(isinstance(e, (int, float)) for e in b["velocity"])]
        if not cands or max(cands) <= 0:
            raise ValueError("da reference_velocity_m_s: no se puede deducir de las fronteras (velocidades dadas como expresión)")
        U, u_def = max(cands), "mayor velocidad media de entrada / corriente libre"
    else:
        U, u_def = _num(U, "reference_velocity_m_s"), "dada por el usuario"
    Lr = reference_length_m
    if Lr is None:
        def immersed(o):                                              # not a plate / step hanging from the outer boundary
            bx = o["box"] if o["type"] == "rectangle" else [o["center"][0] - o["radius"], o["center"][1] - o["radius"],
                                                            o["center"][0] + o["radius"], o["center"][1] + o["radius"]]
            return bx[0] > bbox[0] and bx[2] < bbox[2] and bx[3] < bbox[3] and (bx[1] > bbox[1] or axi)
        free = [o for o in obstacles if immersed(o)]
        if free:
            o = free[0]
            if o["type"] == "circle":
                Lr = 2 * o["radius"]
            else:
                bx = o["box"]
                Lr = 2 * bx[3] if axi else ((bx[3] - bx[1]) if sax == 0 else (bx[2] - bx[0]))
            l_def = f"tamaño transversal del obstáculo '{o['name']}' (diámetro)"
        else:
            ins = [b for b in bnds if b["type"] == "inlet"]
            if ins:
                lo, hi = ins[0]["range"]
                on_axis = axi and ins[0]["axis"] == "x" and abs(lo) < 1e-12
                Lr, l_def = (2 * hi, f"diámetro del tubo en la entrada '{ins[0]['name']}'") if on_axis else \
                    (hi - lo, f"ancho de la entrada '{ins[0]['name']}'")
            else:
                Lr, l_def = min(min(r[2] - r[0], r[3] - r[1]) for r in rects), "lado corto del rectángulo más estrecho"
    else:
        Lr, l_def = _num(Lr, "reference_length_m"), "dada por el usuario"
    if U <= 0 or Lr <= 0:
        raise ValueError("reference_velocity_m_s y reference_length_m deben ser positivos")
    rheo = None
    if rheology is not None:
        if reynolds is not None or viscosity_Pa_s is not None or density_kg_m3 is None:
            raise ValueError("rheology: da density_kg_m3 y los parámetros del modelo en SI; no se admite reynolds ni viscosity_Pa_s "
                             "(la viscosidad la da el modelo)")
        if transient is not None:
            raise ValueError("rheology con transient no está soportado (el transitorio no newtoniano no está validado)")
        rheo = _rheology(rheology, U / Lr)
        rho, mu = float(density_kg_m3), apparent_viscosity(rheo, U / Lr)
        fluid = (f"no newtoniano ({rheo['model']}); la viscosidad de referencia μ_ref = η(γ̇_ref) = {mu:.4g} Pa·s se evalúa en "
                 f"γ̇_ref = U_ref/L_ref = {U / Lr:.4g} 1/s y solo fija la escala de Re")
    elif reynolds is not None:
        if viscosity_Pa_s is not None:
            raise ValueError("da reynolds o viscosity_Pa_s, no ambos")
        rho = density_kg_m3 if density_kg_m3 is not None else 1.0
        mu = rho * U * Lr / _num(reynolds, "reynolds")
        fluid = f"forma adimensional: Re = {reynolds:g} sobre L_ref y U_ref; ρ = {rho:g}, μ = ρ·U_ref·L_ref/Re"
    else:
        if density_kg_m3 is None or viscosity_Pa_s is None:
            raise ValueError("da density_kg_m3 y viscosity_Pa_s, o reynolds (basado en reference_length_m y reference_velocity_m_s)")
        rho, mu, fluid = float(density_kg_m3), float(viscosity_Pa_s), "ρ y μ del usuario (SI)"
    if rho <= 0 or mu <= 0:
        raise ValueError("densidad, viscosidad y Reynolds deben ser positivos")

    # mesh
    m = dict(mesh or {})
    unknown = set(m) - {"size", "obstacle_size", "growth", "boxes", "refine_ratio", "refinement_check", "tolerance"}
    if unknown:
        raise ValueError(f"mesh: claves no reconocidas {sorted(unknown)}")
    area = sum((r[2] - r[0]) * (r[3] - r[1]) for r in rects)
    notes = []
    if m.get("size") is None:
        h = min(min(min(r[2] - r[0], r[3] - r[1]) for r in rects) / 8.0, max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 40.0)
        if not obstacles and area / (0.433 * h * h) > MAX_BASE_CELLS:
            h = math.sqrt(area / (0.433 * MAX_BASE_CELLS))
            notes.append(f"mesh.size por omisión limitado a {h:.3g} para no pasar de ~{MAX_BASE_CELLS} celdas en la malla base")
    else:
        h = _num(m["size"], "mesh.size")
    boxes = []
    for bx in m.get("boxes") or []:
        boxes.append({"box": _box(bx.get("box"), "mesh.boxes[].box"), "size": _num(bx.get("size"), "mesh.boxes[].size")})
    if transient is not None and not boxes and m.get("size") is None:
        # automatic wake box for a transient: the default mesh grows away from the obstacle and would smear the vortices
        for o in obstacles:
            d = o["_size"]
            c = o["center"] if o["type"] == "circle" else [0.5 * (o["box"][0] + o["box"][2]), 0.5 * (o["box"][1] + o["box"][3])]
            lo, hi = [c[0] - 1.2 * d, c[1] - 1.2 * d], [c[0] + 1.2 * d, c[1] + 1.2 * d]
            sgn = 1.0 if stream_direction[0] == "+" else -1.0
            if sgn > 0:
                hi[sax] = c[sax] + 10 * d
            else:
                lo[sax] = c[sax] - 10 * d
            bx = [max(lo[0], bbox[0]), max(lo[1], bbox[1]), min(hi[0], bbox[2]), min(hi[1], bbox[3])]
            if bx[2] > bx[0] and bx[3] > bx[1] and d / 7.0 < h:
                boxes.append({"box": bx, "size": d / 7.0})
                notes.append(f"transient sin mesh.boxes: se añadió una caja de estela de tamaño d/7 = {d / 7.0:.3g} detrás de '{o['name']}' "
                             f"({bx[0]:.3g}..{bx[2]:.3g} × {bx[1]:.3g}..{bx[3]:.3g})")
    for o in obstacles:
        o["mesh_size"] = min(float(m.get("obstacle_size") or o["_size"] / 24.0), h)
        del o["_size"]
    mp = {"size": h, "growth": float(m.get("growth", 0.12)), "boxes": boxes, "refine_ratio": float(m.get("refine_ratio", 1.5)),
          "refinement_check": bool(m.get("refinement_check", True))}
    if h <= 0 or not 1.2 <= mp["refine_ratio"] <= 3 or not 0.01 <= mp["growth"] <= 1:
        raise ValueError("mesh: size > 0, 1.2 ≤ refine_ratio ≤ 3 y 0.01 ≤ growth ≤ 1")

    payload = {"mode": geometry["mode"], "rectangles": rects, "obstacles": obstacles, "boundaries": bnds, "rho": rho, "mu": mu,
               "L_ref": Lr, "U_ref": U, "stream_direction": stream_direction, "mesh": mp}
    if rheo:
        payload["rheology"] = rheo
    if points:
        payload["points"] = [[_num(pt[0], "points"), _num(pt[1], "points")] for pt in points]
    secs = []
    for i, s in enumerate(sections or []):
        ln = _line({k: v for k, v in s.items() if k != "name"}, bbox, rects, f"sections[{i}]")
        secs.append({"name": str(s.get("name") or f"s{i + 1}"), **ln})
    if secs:
        payload["sections"] = secs
    flowb = {b["name"] for b in bnds if b.get("axis")}
    if pressure_drop_between is None:
        pairs = [[a["name"], b["name"]] for a in bnds if a["type"] == "inlet" for b in bnds if b["type"] == "outlet"]
    else:
        pairs = [list(pr) for pr in pressure_drop_between]
        for pr in pairs:
            if len(pr) != 2 or pr[0] not in flowb or pr[1] not in flowb:
                raise ValueError(f"pressure_drop_between: {pr!r} debe ser un par de nombres de fronteras rectas definidas en boundaries")
    payload["pressure_drops"] = pairs
    if scalar:
        if (scalar.get("peclet") is None) == (scalar.get("diffusivity_m2_s") is None):
            raise ValueError("scalar: da peclet (= U_ref·L_ref/D) o diffusivity_m2_s, uno de los dos")
        pe = _num(scalar["peclet"], "scalar.peclet") if scalar.get("peclet") is not None else U * Lr / _num(scalar["diffusivity_m2_s"], "scalar.diffusivity_m2_s")
        vals = scalar.get("values") or {}
        if pe <= 0 or not vals or any(k not in names for k in vals):
            raise ValueError(f"scalar: peclet > 0 y values {{frontera u obstáculo: valor}} con nombres de {names}")
        payload["scalar"] = {"peclet": pe, "values": {k: _num(v, "scalar.values") for k, v in vals.items()}}
        if scalar.get("reference_inlet_value") is not None:
            payload["scalar"]["reference_inlet_value"] = _num(scalar["reference_inlet_value"], "scalar.reference_inlet_value")
    if transient is not None:
        if not isinstance(transient, dict):
            raise ValueError("transient debe ser un objeto ({} = valores por omisión)")
        tref = Lr / U
        unknown = set(transient) - {"dt_s", "t_end_s", "perturbation", "min_cycles", "stop_when_settled", "mesh", "initial",
                                    "refinement", "tolerance"}
        if unknown:
            raise ValueError(f"transient: claves no reconocidas {sorted(unknown)}")
        tr = {"dt": float(transient["dt_s"]) if transient.get("dt_s") else None,       # None: chosen by the solver (see auto_dt)
              "t_end": float(transient.get("t_end_s") or 150 * tref),
              "perturbation": float(transient.get("perturbation", 0.2)), "min_cycles": int(transient.get("min_cycles", 8)),
              "stop_when_settled": bool(transient.get("stop_when_settled", True)), "mesh": transient.get("mesh", "base"),
              "initial": transient.get("initial", "steady"), "refinement": transient.get("refinement", "dt"),
              "tolerance": float(transient.get("tolerance", m.get("tolerance", MESH_TOL)))}
        if tr["initial"] not in ("steady", "rest") or tr["refinement"] not in ("dt", "mesh", "both", "none"):
            raise ValueError("transient: initial debe ser 'steady' o 'rest'; refinement 'dt', 'mesh', 'both' o 'none'")
        if (tr["dt"] is not None and (tr["dt"] <= 0 or tr["t_end"] <= tr["dt"] or tr["t_end"] / tr["dt"] > 200000)) \
                or tr["t_end"] <= 0 or tr["mesh"] not in ("base", "fine") or tr["min_cycles"] < 4 or tr["tolerance"] <= 0:
            raise ValueError("transient: dt_s > 0, t_end_s > dt_s, mesh 'base' o 'fine', min_cycles ≥ 4 y como mucho 200 000 pasos")
        payload["transient"] = tr
    ctx = {"u_def": u_def, "l_def": l_def, "fluid": fluid, "notes": notes, "bbox": bbox, "axi": axi, "sax": sax,
           "tol": float(m.get("tolerance", MESH_TOL))}
    return payload, ctx


def _mesh_evidence(q_f: dict, q_c: dict, scales: dict, tol: float) -> tuple[dict, list]:
    table, bad = {}, []
    for k, f in q_f.items():
        c = q_c.get(k)
        if c is None:
            table[k] = {"fine": f, "coarse": None, "change_rel": None, "converged": False}
            bad.append(k)
            continue
        if k == "recirculation.count":
            ok, ch = f == c, abs(f - c)
        else:
            kind = k.split(".")[0].rstrip("0123456789")
            ref = max(abs(f), abs(c))
            if k.endswith((".velocity_x", ".velocity_y")):            # a component is judged against the local speed
                base = k.rsplit(".", 1)[0]
                ref = math.hypot(q_f.get(base + ".velocity_x", 0.0), q_f.get(base + ".velocity_y", 0.0))
            ch = abs(f - c) / max(ref, 1e-3 * scales.get(kind, 0.0), 1e-300)
            ok = ch < tol
        table[k] = {"fine": f, "coarse": c, "change_rel": ch, "converged": bool(ok)}
        if not ok:
            bad.append(k)
    bad += [k for k in q_c if k not in q_f]
    return table, bad


def _domain_report(payload: dict, ctx: dict, res: dict) -> tuple[dict, list]:
    """Blockage and distances of each obstacle to the domain limits, in obstacle sizes; warnings for unbounded-stream use."""
    rep, warns = {}, []
    bb, axi, sax = ctx["bbox"], ctx["axi"], ctx["sax"]
    free = any(b["type"] in ("stream", "symmetry") for b in payload["boundaries"])
    for o in payload["obstacles"]:
        bx = [o["center"][0] - o["radius"], o["center"][1] - o["radius"], o["center"][0] + o["radius"], o["center"][1] + o["radius"]] \
            if o["type"] == "circle" else o["box"]
        d = (2 * min(bx[3], bb[3])) if axi else (bx[3 - sax] - bx[1 - sax])
        tr0, tr1 = (bb[1], bb[3]) if sax == 0 else (bb[0], bb[2])
        e = {"obstacle_size": d,
             "blockage_ratio": (min(bx[3], bb[3]) ** 2 - max(bx[1], 0.0) ** 2) / bb[3] ** 2 if axi else d / (tr1 - tr0),
             "upstream_length_over_size": (bx[sax] - bb[sax]) / d, "downstream_length_over_size": (bb[sax + 2] - bx[sax + 2]) / d,
             "lateral_half_width_over_size": bb[3] / d if axi else min(bx[1 - sax] - tr0, tr1 - bx[3 - sax]) / d + 0.5,
             "lateral_boundaries": "corriente libre / simetría" if free else "paredes"}
        if axi:
            e["radius_ratio_obstacle_to_domain"] = bx[3] / bb[3]
        rep[o["name"]] = e
        ob = (res or {}).get("obstacles", {}).get(o["name"])
        re_o = ob["reynolds_obstacle"] if ob else None
        if free and (e["blockage_ratio"] > 0.05 or (re_o is not None and re_o < 5)):
            warns.append(f"Obstáculo '{o['name']}' en corriente 'no confinada' simulada en un dominio finito (bloqueo "
                         f"{e['blockage_ratio']:.3g}, {e['upstream_length_over_size']:.0f} tamaños aguas arriba, "
                         f"{e['lateral_half_width_over_size']:.0f} al lado): a Re bajo el arrastre depende del tamaño del dominio "
                         "(esfera: factor ≈ 1/(1 − 2.1·a/R); cilindro 2-D: no existe límite de Stokes, paradoja de Stokes). "
                         "Repite con un dominio mayor para medir esa dependencia.")
    return rep, warns


def laminar_flow_2d(geometry: dict, boundaries: list, density_kg_m3: float | None = None, viscosity_Pa_s: float | None = None,
                    reynolds: float | None = None, reference_length_m: float | None = None,
                    reference_velocity_m_s: float | None = None, stream_direction: str = "+x", points: list | None = None,
                    sections: list | None = None, pressure_drop_between: list | None = None, scalar: dict | None = None,
                    transient: dict | None = None, mesh: dict | None = None, figure_path: str | None = None,
                    timeout_s: int | None = None, rheology: dict | None = None, max_wall_s: float | None = None) -> dict:
    """Steady (optionally transient) laminar incompressible flow on a 2-D planar or axisymmetric geometry with FEniCSx
    (separate process). See the module docstring for the evidence and `packages.dwsim_client.tools_flow2d` for the
    user-facing description of every argument."""
    payload, ctx = _prepare(geometry, boundaries, density_kg_m3, viscosity_Pa_s, reynolds, reference_length_m,
                            reference_velocity_m_s, stream_direction, points, sections, pressure_drop_between, scalar,
                            transient, mesh, rheology)
    if max_wall_s is not None:
        max_wall_s = _num(max_wall_s, "max_wall_s")
        if max_wall_s <= 0:
            raise ValueError("max_wall_s debe ser positivo (segundos de pared para el transitorio)")
    if timeout_s is None:
        timeout_s = 1800 if transient is not None else 300
        if max_wall_s is not None:                                    # room to finish the step, post-process and write
            timeout_s = int(max_wall_s + max(120.0, 0.25 * max_wall_s))
    # wall-clock budget of the solver process: the transient stops cleanly there and returns a PARTIAL result
    payload["time_budget_s"] = max(timeout_s - 20, 0.6 * timeout_s) if max_wall_s is None else min(max_wall_s, max(timeout_s - 5, 0.5 * timeout_s))
    if figure_path:
        payload["figure_path"] = str(Path(figure_path).expanduser().resolve())
    L, U, rho, mu = payload["L_ref"], payload["U_ref"], payload["rho"], payload["mu"]
    Re = rho * U * L / mu
    if figure_path:
        payload["figure_title"] = f"Flujo laminar 2-D ({'axisimétrico' if ctx['axi'] else 'plano'}) · Re = {Re:.3g} · FEniCSx"
    try:
        out = RF.run_solver(payload, timeout_s=timeout_s, solver=SOLVER)
    except RuntimeError as ex:
        if "no terminó" in str(ex):
            raise RuntimeError(f"FEniCSx no terminó en {timeout_s} s (laminar_flow_2d). Sube timeout_s, engruesa la malla "
                               "(mesh.size, mesh.obstacle_size), acorta el transitorio (t_end_s, dt_s) o da max_wall_s menor que "
                               "timeout_s para recibir un resultado parcial.") from None
        raise
    res = out.get("result")
    newton = out["steady"]["newton"]
    warnings, reasons = list(ctx["notes"]), []
    ev = {"newton_converged": bool(out["steady"]["converged"]), "newton_residual_norm": newton["residual_norm"],
          "newton_residual_norm_initial": newton.get("residual_norm_initial"),
          "newton_iterations": newton["newton_iterations"], "continuation_in_Re": bool(newton.get("continuation_used")),
          "newton_path": newton["path"], "newton_tolerances": newton.get("tolerances")}
    rheo = payload.get("rheology")
    if rheo:
        ev["continuation_in_rheology"] = bool(newton.get("continuation_in_rheology_used"))
    if not ev["newton_converged"] and rheo and newton.get("theta_reached") is not None:
        reasons.append(f"Newton no convergió con el modelo reológico completo (la continuación llegó a θ = {newton['theta_reached']:.3g} "
                       "de 1 entre newtoniano y el modelo): sube rheology.gamma_reg_1_s o refina la malla")
    elif not ev["newton_converged"]:
        reasons.append(f"Newton no convergió al Re pedido (alcanzó Re = {newton.get('Re_reached', 0):.4g} por continuación): "
                       "probablemente no hay solución estacionaria estable alcanzable; usa transient o refina la malla")
    result = {"converged": ev["newton_converged"], "mode": payload["mode"], "Re": Re,
              "reference": {"length_m": L, "length_definition": ctx["l_def"], "velocity_m_s": U,
                            "velocity_definition": ctx["u_def"], "time_s": L / U, "density_kg_m3": rho, "viscosity_Pa_s": mu,
                            "fluid": ctx["fluid"]},
              "units": ("SI. Caudales y fuerzas " + ("totales (integrados en 2πr): m³/s y N" if ctx["axi"]
                        else "POR UNIDAD DE PROFUNDIDAD: m²/s y N/m") + ". Si se dio reynolds sin fluido, ρ = 1 y las "
                        "presiones y fuerzas quedan en esas unidades (úsense los coeficientes adimensionales)."),
              "stream_direction": stream_direction}
    if rheo:
        rate = U / L
        rb = {"model": rheo["model"], "parameters": {k: v for k, v in rheo.items() if k not in ("model", "gamma_reg_1_s", "newtonian_equivalent_Pa_s")},
              "viscosity_law": {"power_law": "η = K·(γ̇² + γ̇_reg²)^((n−1)/2)",
                                "carreau": "η = η∞ + (η0 − η∞)·[1 + (λγ̇)²]^((n−1)/2)",
                                "carreau_yasuda": "η = η∞ + (η0 − η∞)·[1 + (λγ̇_r)^a]^((n−1)/a), γ̇_r = √(γ̇² + γ̇_reg²)"}[rheo["model"]],
              "shear_rate_definition": "γ̇ = √(2 D:D), D = (∇u + ∇uᵀ)/2" + ("; axisimétrico: incluye D_θθ = u_r/r" if ctx["axi"] else ""),
              "regularisation_shear_rate_1_s": rheo.get("gamma_reg_1_s"),
              "regularisation_note": ("la ley de potencia se regulariza cerca de γ̇ = 0: η ≤ K·γ̇_reg^(n−1)" if rheo["model"] == "power_law"
                                      else ("γ̇_reg solo hace derivable (λγ̇)^a en γ̇ = 0" if rheo["model"] == "carreau_yasuda"
                                            else "sin regularización: Carreau es regular en γ̇ = 0")),
              "reference_shear_rate_1_s": rate, "reference_viscosity_Pa_s": mu,
              "Re_definition": "Re = ρ·U_ref·L_ref/η(γ̇_ref), γ̇_ref = U_ref/L_ref (escala, no un Re generalizado)",
              "iteration": "Newton (SNES, búsqueda lineal) sobre el sistema completo con η(γ̇) derivada exactamente; arranque desde la "
                           "solución newtoniana con μ_ref y, si falla, continuación η_θ = μ_ref^(1−θ)·η^θ",
              "newtonian_equivalent_Pa_s": rheo.get("newtonian_equivalent_Pa_s")}
        if rheo["model"] == "power_law" and ctx["axi"] and "diámetro del tubo" in ctx["l_def"]:
            n_, K_ = rheo["n"], rheo["K"]
            # Metzner & Reed (1955) generalised Reynolds number for a power-law fluid in a tube (definition)
            rb["Re_metzner_reed"] = rho * U ** (2 - n_) * L ** n_ / (K_ * 8 ** (n_ - 1) * ((3 * n_ + 1) / (4 * n_)) ** n_)
            rb["Re_metzner_reed_definition"] = "ρ·ū^(2−n)·D^n / [K·8^(n−1)·((3n+1)/(4n))^n], con ū = U_ref y D = L_ref"
        result["rheology"] = rb
    if res is not None and (ev["newton_converged"] or transient is not None):
        for k in ("boundaries", "mass_balance", "pressure_drops", "points", "sections", "straight_sections", "obstacles",
                  "recirculation", "scalar", "velocity_max", "pressure_min_max"):
            if k in res:
                result[k] = res[k]
        if rheo and res.get("rheology_field"):
            result["rheology"]["field"] = res["rheology_field"]
            frac = res["rheology_field"]["cells_below_regularisation_rate_fraction"]
            if rheo["model"] == "power_law" and frac > 0.05:
                warnings.append(f"Ley de potencia: en el {frac:.0%} de las celdas γ̇ < γ̇_reg = {rheo['gamma_reg_1_s']:.3g} 1/s (viscosidad "
                                "regularizada, no la del modelo); baja rheology.gamma_reg_1_s y comprueba que el resultado no cambia.")
        mb = res["mass_balance"]
        ev["mass_imbalance_rel"] = mb["imbalance_rel"]
        ev["mass_balanced"] = mb["imbalance_rel"] < MASS_TOL
        if not ev["mass_balanced"]:
            reasons.append(f"desbalance de masa relativo {mb['imbalance_rel']:.2e}")
        # mesh refinement
        if "quantities_coarse" in out:
            flow_scale = max(mb["flow_in"], mb["flow_out"], U * L ** (2 if ctx["axi"] else 1) * 1e-6)
            pmax = max(abs(v) for v in res["pressure_min_max"])
            scales = {"flow": flow_scale, "section": flow_scale, "mean_pressure": pmax, "pressure_drop": pmax, "point": res["velocity_max"],
                      "recirculation": L, "straight_section": pmax / L}
            table, bad = _mesh_evidence(out["quantities"], out["quantities_coarse"], scales, ctx["tol"])
            ev.update(mesh_refinement=table, mesh_converged=not bad, not_mesh_converged=bad, mesh_tolerance_rel=ctx["tol"],
                      meshes=out["meshes"])
            if bad:
                reasons.append("no convergidos en malla (cambio > {:.0%} entre las dos mallas): {}; reduce mesh.size / "
                               "mesh.obstacle_size o añade mesh.boxes".format(ctx["tol"], ", ".join(bad)))
        else:
            ev.update(mesh_converged=None, meshes=out["meshes"])
            reasons.append("sin estudio de malla (mesh.refinement_check = false)")
        # Poiseuille self-check on developed straight sections
        dev = [s for s in res["straight_sections"] if s["developed"] and s.get("poiseuille_error_rel") is not None]
        if dev:
            worst = max(abs(s["poiseuille_error_rel"]) for s in dev)
            ev["poiseuille_self_check"] = worst < POISEUILLE_TOL
            ev["poiseuille_error_rel"] = worst
            ev["poiseuille_sections_checked"] = len(dev)
            if not ev["poiseuille_self_check"]:
                reasons.append(f"un tramo recto desarrollado no reproduce Poiseuille (error {worst:.2%})")
        else:
            ev["poiseuille_self_check"] = None
            ev["poiseuille_note"] = ("no hay tramo recto entre paredes con flujo desarrollado en esta geometría"
                                     if not res["straight_sections"] else "ningún tramo recto llega a flujo desarrollado (son cortos para este Re)")
        for name, ob in res.get("obstacles", {}).items():
            if ob["drag"]:
                ob["drag_direct_vs_consistent_rel"] = abs(ob["drag_direct_stress_integral"] - ob["drag"]) / abs(ob["drag"])
        dom, w2 = _domain_report(payload, ctx, res)
        if dom:
            result["domain"] = dom
            warnings += w2
        if res.get("scalar") and res["scalar"].get("cell_peclet_max_estimate", 0) > 50:
            warnings.append(f"Escalar: Péclet de celda ≈ {res['scalar']['cell_peclet_max_estimate']:.0f}; la capa límite de "
                            "concentración puede no estar resuelta (SUPG la estabiliza, no la resuelve): mira el cambio de Nu/Sh con la malla.")
        # regime warnings
        for name, ob in res.get("obstacles", {}).items():
            o = next(o for o in payload["obstacles"] if o["name"] == name)
            re_o = Re * ob["characteristic_length"] / L
            ob["reynolds_obstacle"] = re_o
            if transient is None and not rheo and o["type"] == "circle":
                lim = 210.0 if ctx["axi"] else 47.0
                if re_o > lim:
                    warnings.append(f"Re del obstáculo '{name}' = {re_o:.0f} > {lim:.0f}: la solución ESTACIONARIA calculada existe pero es "
                                    + ("inestable (la estela de una esfera deja de ser axisimétrica; fuera del alcance 2-D)." if ctx["axi"]
                                       else "inestable; el flujo real desprende vórtices. Usa transient para St y el C_D medio."))
    elif res is None:
        ev.update(mesh_converged=None, mass_balanced=None, poiseuille_self_check=None, meshes=out["meshes"])
    if out.get("default_wall_facets", 0) and not any(b["type"] == "outlet" for b in payload["boundaries"]):
        warnings.append("Sin frontera 'outlet': la presión queda definida salvo una constante; se reporta con media nula (solo valen diferencias).")
    if Re > 2300 and not payload["obstacles"]:
        warnings.append(f"Re = {Re:.0f} sobre L_ref: en un conducto el flujo real sería de transición o turbulento; este modelo es laminar.")
    if transient is not None:
        tr = out.get("transient", {})
        result["transient"] = tr
        ev["transient_state"] = tr.get("state")
        ev["transient_complete"] = bool(tr.get("complete"))
        checks = tr.get("refinement", {})
        ev["transient_refinement"] = {k: {kk: c.get(kk) for kk in ("what", "done", "converged", "why", "table", "tolerance_rel")}
                                      for k, c in checks.items()}
        dt_ok = bool(checks.get("time_step", {}).get("done") and checks["time_step"].get("converged"))
        mesh_ok = bool(checks.get("mesh", {}).get("done") and checks["mesh"].get("converged"))
        ev["transient_time_step_converged"] = dt_ok if "time_step" in checks else None
        ev["transient_mesh_converged"] = mesh_ok if "mesh" in checks else None
        ev["transient_refinement_checked"] = bool(dt_ok and mesh_ok)
        if not tr.get("complete"):
            result["partial"] = True
            cut = tr.get("stopped_by") == "time_budget"
            result["status"] = ("RESULTADO PARCIAL, NO TERMINADO: el transitorio " +
                                (f"se cortó por el presupuesto de tiempo de pared ({payload['time_budget_s']:.0f} s)" if cut
                                 else f"paró por {tr.get('stopped_by')} en estado {tr.get('state')}") +
                                f" a t = {tr.get('time_simulated', 0):.4g} s ({tr.get('steps', 0)} pasos). Los números de "
                                "transient.provisional_not_final NO son valores finales y no deben citarse como resultado.")
            reasons.append(result["status"])
            if tr.get("state") == "diverged":
                reasons.append("el transitorio divergió: reduce dt_s o alarga el dominio aguas abajo")
            elif cut:
                reasons.append("sube max_wall_s / timeout_s, engruesa la malla o da dt_s mayor para terminar")
            else:
                reasons.append("el transitorio no llegó a un régimen establecido: sube t_end_s")
        else:
            result["partial"] = False
        for key, label, hint in (("time_step", "paso de tiempo", "reduce dt_s"), ("mesh", "malla", "reduce mesh.size / mesh.obstacle_size")):
            c = checks.get(key)
            if c is None:
                reasons.append(f"transitorio sin estudio de {label} (transient.refinement = '{payload['transient']['refinement']}'; "
                               "usa 'both' para comprobar paso de tiempo y malla)")
            elif not c.get("done"):
                reasons.append(f"transitorio: la comprobación de {label} no se completó ({c.get('why')})")
            elif not c.get("converged"):
                bad = [k for k, v in (c.get("table") or {}).items() if not v["converged"]]
                reasons.append(f"transitorio no convergido en {label}: {', '.join(bad)} cambian más de {c.get('tolerance_rel', 0):.0%}; {hint}")
        spp = tr.get("steps_per_period") or tr.get("provisional_not_final", {}).get("steps_per_period")
        if spp and spp < 40:
            warnings.append(f"Solo {spp:.0f} pasos por periodo: reduce dt_s (BDF2 amortigua y desfasa con pocos pasos).")
        if ctx["axi"]:
            warnings.append("Transitorio axisimétrico: no captura inestabilidades tridimensionales (estela de esfera a Re > ~210).")
    ev["verified"] = not reasons
    ev["reasons"] = reasons
    result.update(mesh=out["steady"]["mesh"], evidence=ev, warnings=warnings, timing_s=out["timing_s"],
                  figure_path=out.get("figure_path"), engine=f"FEniCSx dolfinx {out.get('dolfinx_version', '?')} + gmsh",
                  assumptions=[("fluido newtoniano generalizado (" + rheo["model"] + ": viscosidad aparente función de γ̇; forma "
                                "(2η D(u), D(v)) del término viscoso)" if rheo else "fluido newtoniano") +
                               " incompresible de propiedades constantes, flujo laminar, sin gravedad (p es la presión "
                               "dinámica: añádase ρgh aparte)",
                               "2-D plano (profundidad infinita) o axisimétrico sin giro (x = eje z, y = radio r)",
                               "Taylor-Hood P2-P1, Newton con LU directo (MUMPS); salida p = p0 con flujo normal",
                               "fuerzas por reacción del residuo (consistente); parte de presión = ∫ p n dS; viscosa = total − presión"],
                  not_included=["turbulencia", "flujo 3-D o con giro", "superficie libre, dos fases",
                                "esfuerzo de cedencia (Bingham, Herschel-Bulkley), viscoelasticidad, tixotropía; transitorio no newtoniano",
                                "propiedades variables / flotación", "paredes curvas distintas de círculos, o inclinadas",
                                "escalar transitorio o acoplado al flujo"])
    return result
