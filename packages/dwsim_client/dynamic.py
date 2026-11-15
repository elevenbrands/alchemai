"""Dynamic simulation of a flowsheet with DWSIM's own dynamic mode, driven headless through pythonnet.

DWSIM is GPL-3.0. This module only *calls* its public API (Fluent API `Flowsheet.Dynamics` / `RunDynamics`, which wraps
`DWSIM.Automation.DynamicRunner.IntegratorRunner`) from a separate Python process; nothing from DWSIM is copied here and
no physics is re-implemented: every number in a time series is computed by DWSIM.

What DWSIM does in dynamic mode (read in its source, confirmed numerically here):
  * Sequential-modular explicit Euler. At every step each holdup unit adds `inlet flow x step` and subtracts
    `outlet flow of the previous step x step` from its inventory, flashes the content and publishes level and outlet
    pressure; pressure-flow elements (valves with a Kv) then recompute the flows from the pressures on both sides.
  * Every stream carries a dynamic specification, "flow" or "pressure". A tank integrates its holdup only when its
    outlet is pressure-specified; a Kv valve between two pressure-specified streams computes its own flow.
  * Open tank: vapour space fixed at 101325 Pa, outlet pressure = 101325 + rho_liquid * 9.8 * level (9.8 is DWSIM's).
  * Liquid valve: W[kg/h] = Kv(opening) * sqrt(rho[kg/m3] * 999.1 * dP[bar]), choked at dP = FL^2 (P1 - FF Pv), FL = 0.9.
  * DWSIM labels the sample taken after the first step as t = 0. Here t = 0 is the initial condition (read after a
    priming step of 1e-6 s that only makes levels, pressures and flows consistent) and the n-th solved step is reported
    at t = n * step; an event asked at time t acts on the steps that start at t or later.

What is the caller's data and is never invented: compounds, property package, feeds, boundary pressures, vessel volume
and height, initial level and content, valve Kv, characteristic and opening, controller settings, step, horizon, events.

Evidence of each run (numerics of this run, not the caller's data): the steady-state initialisation solved, the
integration finished without exceptions, the mass balance of every tank closes with the explicit rule DWSIM integrates
with (accumulation = sum of in - out), no tank ran empty or over its height (DWSIM's tank has neither a low-level cut
nor an overflow), and the series do not move when the step is halved (the whole case is run again at step / 2).

Env: see `packages.dwsim_client.fluent` (DWSIM_BIN, DOTNET_ROOT).
"""
from __future__ import annotations

import math
from typing import Any

PRIMING_STEP_S = 1.0e-6          # step of the priming pass (numerical; moves each inventory by flow x 1e-6 s)
MASS_BALANCE_TOL = 1.0e-6        # relative closure of accumulation = in - out with DWSIM's own explicit rule
STEP_TOL_DEFAULT = 0.01          # time-averaged change of a series when the step is halved, relative to its range
SPAN_FLOOR_REL = 1.0e-4          # a series that moves less than this fraction of its size counts as constant (flash noise)
SETTLE_TOL_DEFAULT = 1.0e-3      # change over the last fifth of the settling window, relative to the value
MAX_STEPS = 50000                # hard limit of solved steps of the main run (the check run solves twice as many)
P_ATM_DWSIM = 101325.0           # vapour-space pressure DWSIM's open tank is hard-wired to
G_DWSIM = 9.8                    # gravity constant inside DWSIM's tank model (not 9.80665)

UNIT_KINDS = ("tank", "valve", "pid")
VALVE_CHARACTERISTICS = ("linear", "equal_percentage", "quick_opening")

# alias -> (DWSIM property id, units asked to DWSIM, writable by an event or a controller)
STREAM_VARIABLES = {
    "temperature_K": ("PROP_MS_0", "K", True), "pressure_Pa": ("PROP_MS_1", "Pa", True),
    "mass_flow_kg_s": ("PROP_MS_2", "kg/s", True), "molar_flow_mol_s": ("PROP_MS_3", "mol/s", True),
    "density_kg_m3": ("PROP_MS_5", "kg/m3", False), "specific_enthalpy_kJ_kg": ("PROP_MS_7", "kJ/kg", False)}
STREAM_COMPOUND_VARIABLES = {"mole_fraction": "PROP_MS_102", "mass_fraction": "PROP_MS_103"}
UNIT_VARIABLES = {
    "tank": {"level_m": ("Liquid Level", "m", False)},
    "valve": {"opening_pct": ("PROP_VA_5", "", True), "pressure_drop_Pa": ("PROP_VA_1", "Pa", False)},
    "pid": {"setpoint": ("SetPointAbs", "", True), "output": ("OutputAbs", "", False)}}

NOT_INCLUDED = [
    "Solo se arman tanques abiertos (a la atmósfera), válvulas de líquido con Kv y controladores PID; DWSIM tiene modelo "
    "dinámico de más equipos (separadores, bombas, compresores, calentadores, intercambiadores, CSTR, columnas) que aquí "
    "no están cableados ni validados",
    "Tanque cerrado con presión del espacio de vapor, pérdidas de calor al ambiente y rebose: el tanque abierto de DWSIM "
    "no tiene rebose (el nivel sigue subiendo por encima de la altura) ni corte por bajo nivel",
    "Válvulas de gas o bifásicas, retardo y constante de tiempo del actuador, cavitación",
    "Rampas y otros perfiles de evento: solo cambios escalón (una rampa se da como varios escalones)",
    "Integrador implícito y paso adaptativo de DWSIM: se usa Euler explícito de paso fijo",
    "Hidráulica de tuberías entre equipos, inercia del fluido y tiempos muertos de transporte",
]


def available() -> tuple[bool, str]:
    """(True, "") when DWSIM and its dynamics API can be loaded in this process, else (False, reason)."""
    try:
        _net()
        return True, ""
    except Exception as ex:  # noqa: BLE001 - any failure to start .NET or to find the assemblies
        return False, f"{type(ex).__name__}: {(str(ex).splitlines() or [''])[0]}"


def _net() -> dict:
    """Load DWSIM lazily and return the handful of .NET types used here."""
    from packages.dwsim_client.fluent import fluent

    Flowsheet, PropertyPackages, Q = fluent()
    import System  # noqa: F401
    from DWSIM.Automation.FluentAPI.Builders import CompositionBuilder
    from DWSIM.UnitOperations.UnitOperations import Valve

    return {"Flowsheet": Flowsheet, "PP": PropertyPackages, "Q": Q, "System": System, "Valve": Valve,
            "CompositionBuilder": CompositionBuilder}


# ----------------------------------------------------------------------------- validation (pure Python, no DWSIM)
def _need(d: dict, key: str, what: str, purpose: str):
    if key not in d or d[key] is None:
        raise ValueError(f"Falta «{key}» en {what}: {purpose}.")
    return d[key]


def _positive(x: Any, name: str, what: str, allow_zero: bool = False) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"«{name}» de {what} debe ser un número.") from None
    if not math.isfinite(v) or v < 0 or (v == 0 and not allow_zero):
        raise ValueError(f"«{name}» de {what} debe ser {'≥ 0' if allow_zero else '> 0'} (se recibió {x!r}).")
    return v


def _fractions(z: Any, compounds: list[str], what: str) -> dict[str, float]:
    if not isinstance(z, dict) or not z:
        raise ValueError(f"«mole_fractions» de {what} debe ser un diccionario compuesto → fracción molar.")
    unknown = [c for c in z if c not in compounds]
    if unknown:
        raise ValueError(f"«mole_fractions» de {what} nombra compuestos que no están en el flowsheet: {unknown}.")
    out = {c: float(z.get(c, 0.0)) for c in compounds}
    if any(v < 0 for v in out.values()) or abs(sum(out.values()) - 1.0) > 1e-6:
        raise ValueError(f"«mole_fractions» de {what} debe ser ≥ 0 y sumar 1 (suma {sum(out.values()):.6g}).")
    return out


def _variable(kind: str, variable: str, compounds: list[str], what: str) -> tuple[str, str, bool]:
    """(DWSIM property id, units, writable) of an alias for a stream or a unit kind."""
    if kind == "stream":
        if variable in STREAM_VARIABLES:
            return STREAM_VARIABLES[variable]
        head, _, comp = variable.partition(":")
        if head in STREAM_COMPOUND_VARIABLES and comp in compounds:
            return f"{STREAM_COMPOUND_VARIABLES[head]}/{comp}", "", False
        valid = sorted(STREAM_VARIABLES) + [f"{h}:<compuesto>" for h in STREAM_COMPOUND_VARIABLES]
    else:
        if variable in UNIT_VARIABLES[kind]:
            return UNIT_VARIABLES[kind][variable]
        valid = sorted(UNIT_VARIABLES[kind])
    raise ValueError(f"Variable «{variable}» desconocida en {what}; válidas para {kind}: {valid}.")


def _validate_tank(u: dict, name: str, compounds: list[str]) -> None:
    what = f"el equipo «{name}»"
    u["volume_m3"] = _positive(_need(u, "volume_m3", what, "el volumen del tanque, que fija el inventario"),
                               "volume_m3", what)
    u["height_m"] = _positive(_need(u, "height_m", what, "la altura del tanque (área = volumen / altura), que fija el "
                                    "nivel y la presión hidrostática de salida"), "height_m", what)
    u["initial_level_m"] = _positive(_need(u, "initial_level_m", what, "el nivel de líquido en t = 0 (condición inicial)"),
                                     "initial_level_m", what, allow_zero=True)
    if u["initial_level_m"] > u["height_m"]:
        raise ValueError(f"«initial_level_m» de {what} no puede superar «height_m».")
    if u["initial_level_m"] == 0:
        return
    ic = _need(u, "initial_content", what, "con qué está lleno el tanque en t = 0: \"inlet\" (igual que su corriente de "
               "entrada) o {\"temperature_K\", \"mole_fractions\"}")
    if isinstance(ic, dict):
        inside = f"«initial_content» de «{name}»"
        temperature = _positive(_need(ic, "temperature_K", inside, "la temperatura del contenido inicial"),
                                "temperature_K", what)
        if len(compounds) > 1:
            z = _need(ic, "mole_fractions", inside, "la composición del contenido inicial")
        else:
            z = ic.get("mole_fractions") or {compounds[0]: 1.0}
        u["initial_content"] = {"temperature_K": temperature, "mole_fractions": _fractions(z, compounds, what)}
    elif ic != "inlet":
        raise ValueError(f"«initial_content» de {what} debe ser \"inlet\" o un diccionario con temperature_K y "
                         "mole_fractions.")


def _validate_valve(u: dict, name: str) -> None:
    what = f"el equipo «{name}»"
    u["kv"] = _positive(_need(u, "kv", what, "el Kv de la válvula totalmente abierta (m³/h de agua con 1 bar de caída); es "
                              "la relación presión-caudal que DWSIM exige en modo dinámico"), "kv", what)
    u["opening_pct"] = _positive(_need(u, "opening_pct", what, "la apertura en t = 0, de 0 a 100 %"), "opening_pct", what,
                                 allow_zero=True)
    if u["opening_pct"] > 100:
        raise ValueError(f"«opening_pct» de {what} debe estar entre 0 y 100.")
    ch = str(_need(u, "characteristic", what,
                   f"la característica apertura-Kv, una de {list(VALVE_CHARACTERISTICS)}")).lower()
    if ch not in VALVE_CHARACTERISTICS:
        raise ValueError(f"«characteristic» de {what} debe ser una de {list(VALVE_CHARACTERISTICS)}.")
    u["characteristic"] = ch
    if ch == "equal_percentage":
        u["rangeability"] = _positive(_need(u, "rangeability", what, "la rangeabilidad R de la característica "
                                            "isoporcentual, Kv = Kv_max · R^(apertura/100 − 1)"), "rangeability", what)


PID_SETTINGS = (
    ("setpoint", "la consigna, en las unidades SI de la variable medida"),
    ("kp", "la ganancia proporcional (adimensional: actúa sobre el error dividido entre |consigna inicial|)"),
    ("ki", "la ganancia integral en 1/s (0 para control P)"), ("kd", "la ganancia derivativa en s (0 para PI)"),
    ("output_bias", "el valor de la variable manipulada con salida cero del controlador"),
    ("output_span", "el rango de la variable manipulada por unidad de salida del controlador"),
    ("output_min", "el límite inferior de la variable manipulada"),
    ("output_max", "el límite superior de la variable manipulada"))


def _validate_stream(name: str, s: dict, compounds: list[str], U: dict, producer: dict, consumer: dict) -> None:
    what = f"la corriente «{name}»"
    spec = str(_need(s, "dynamic_spec", what, "la especificación dinámica que DWSIM exige en cada corriente: \"flow\" (su "
                     "caudal es dato) o \"pressure\" (su presión es dato o la fija un tanque, y el caudal se calcula)")).lower()
    if spec not in ("flow", "pressure"):
        raise ValueError(f"«dynamic_spec» de {what} debe ser \"flow\" o \"pressure\".")
    s["dynamic_spec"] = spec
    src, dst = producer.get(name), consumer.get(name)
    if src is None and dst is None:
        raise ValueError(f"La corriente «{name}» no está conectada a ningún equipo.")
    if src is None:                                   # boundary inlet
        s["temperature_K"] = _positive(_need(s, "temperature_K", what, "la temperatura de la alimentación"),
                                       "temperature_K", what)
        s["pressure_Pa"] = _positive(_need(s, "pressure_Pa", what, "la presión de la alimentación"), "pressure_Pa", what)
        if s.get("mass_flow_kg_s") is None and s.get("molar_flow_mol_s") is None:
            hint = (" (con dynamic_spec \"pressure\" es solo el valor del arranque estacionario; después lo calcula la "
                    "válvula)") if spec == "pressure" else ""
            raise ValueError(f"Falta el caudal de {what}: «mass_flow_kg_s» o «molar_flow_mol_s»{hint}.")
        key = "mass_flow_kg_s" if s.get("mass_flow_kg_s") is not None else "molar_flow_mol_s"
        s[key] = _positive(s[key], key, what)
        if len(compounds) > 1 or s.get("mole_fractions") is not None:
            s["mole_fractions"] = _fractions(_need(s, "mole_fractions", what, "la composición de la alimentación"),
                                             compounds, what)
        else:
            s["mole_fractions"] = {compounds[0]: 1.0}
        if U[dst]["kind"] == "tank" and spec != "flow":
            raise ValueError(f"La corriente «{name}» alimenta un tanque desde el límite: debe ser dynamic_spec \"flow\" (o "
                             "pasar antes por una válvula con Kv).")
    elif U[src]["kind"] == "tank":
        if spec != "pressure":
            raise ValueError(f"La corriente «{name}» sale del tanque «{src}»: debe ser dynamic_spec \"pressure\" (DWSIM "
                             "solo integra el inventario del tanque así; la presión la fija el nivel).")
        if dst is None or U[dst]["kind"] != "valve":
            raise ValueError(f"La salida «{name}» del tanque «{src}» debe ir a una válvula con Kv: sin relación "
                             "presión-caudal aguas abajo DWSIM no puede calcular el caudal de salida.")
    else:                                             # produced by a valve
        if spec != "pressure":
            raise ValueError(f"La corriente «{name}» sale de la válvula «{src}»: debe ser dynamic_spec \"pressure\" con su "
                             "«pressure_Pa» (la válvula calcula el caudal entre dos presiones).")
        s["pressure_Pa"] = _positive(_need(s, "pressure_Pa", what, f"la presión aguas abajo de la válvula «{src}» "
                                           "(contrapresión o límite de batería; 101325 Pa si descarga a un tanque abierto)"),
                                     "pressure_Pa", what)
        if dst is not None and U[dst]["kind"] == "valve":
            raise ValueError(f"Dos válvulas en serie sin inventario entre ellas («{src}» → «{dst}») no están soportadas.")


def _validate(compounds, property_package, streams, units, step_s, duration_s, monitors, events, settle_s) -> dict:
    """Check the whole specification and return it normalised. Raises ValueError (Spanish) naming what is missing."""
    if not compounds or not all(isinstance(c, str) and c for c in compounds):
        raise ValueError("Falta «compounds»: la lista de compuestos (nombres de DWSIM) del flowsheet.")
    compounds = list(compounds)
    if not property_package:
        raise ValueError("Falta «property_package»: el paquete termodinámico de DWSIM con el que se calculan densidades y "
                         "entalpías.")
    run = "la corrida"
    dt = _positive(_need({"step_s": step_s}, "step_s", run, "el paso de integración en segundos"), "step_s", run)
    horizon = _positive(_need({"duration_s": duration_s}, "duration_s", run, "el horizonte a simular en segundos"),
                        "duration_s", run)
    settle = 0.0 if settle_s is None else _positive(settle_s, "settle_s", run, allow_zero=True)
    if not streams:
        raise ValueError("Falta «streams»: las corrientes de materia del flowsheet.")
    if not units:
        raise ValueError("Falta «units»: los equipos del flowsheet (tank, valve, pid).")

    S: dict[str, dict] = {}
    for s in streams:
        name = _need(s, "name", "una corriente", "el nombre con el que los equipos la conectan")
        if name in S:
            raise ValueError(f"La corriente «{name}» está repetida.")
        S[name] = dict(s)
    U: dict[str, dict] = {}
    producer: dict[str, str] = {}
    consumer: dict[str, str] = {}
    for u in units:
        name = _need(u, "name", "un equipo", "el nombre del equipo")
        kind = str(_need(u, "kind", f"el equipo «{name}»", f"el tipo de equipo, uno de {list(UNIT_KINDS)}")).lower()
        if kind not in UNIT_KINDS:
            raise ValueError(f"Tipo de equipo «{kind}» no soportado en «{name}»; soportados: {list(UNIT_KINDS)}.")
        if name in U or name in S:
            raise ValueError(f"El nombre «{name}» está repetido.")
        u = dict(u, kind=kind)
        what = f"el equipo «{name}»"
        if kind in ("tank", "valve"):
            for port, table in (("feed", consumer), ("product", producer)):
                sname = _need(u, port, what, "la corriente conectada a ese puerto")
                if sname not in S:
                    raise ValueError(f"El equipo «{name}» conecta «{port}» a la corriente «{sname}», que no está en "
                                     "«streams».")
                if sname in table:
                    raise ValueError(f"La corriente «{sname}» está conectada dos veces como «{port}».")
                table[sname] = name
        if kind == "tank":
            _validate_tank(u, name, compounds)
        elif kind == "valve":
            _validate_valve(u, name)
        U[name] = u

    def _target(ref: Any, what: str, need_writable: bool) -> dict:
        if not isinstance(ref, dict):
            raise ValueError(f"{what} debe ser un diccionario {{\"object\", \"variable\"}}.")
        obj = _need(ref, "object", what, "el nombre de la corriente o equipo")
        var = _need(ref, "variable", what, "la variable (alias en SI)")
        if obj not in S and obj not in U:
            raise ValueError(f"{what} nombra «{obj}», que no es una corriente ni un equipo del flowsheet.")
        kind = "stream" if obj in S else U[obj]["kind"]
        prop, un, writable = _variable(kind, var, compounds, what)
        if need_writable and not writable:
            raise ValueError(f"La variable «{var}» de «{obj}» no se puede escribir (en {what}).")
        return {"object": obj, "variable": var, "property": prop, "units": un, "label": f"{obj}.{var}"}

    for name, u in U.items():
        if u["kind"] != "pid":
            continue
        what = f"el controlador «{name}»"
        u["controlled"] = _target(_need(u, "controlled", what, "la variable medida"), f"«controlled» de {what}", False)
        u["manipulated"] = _target(_need(u, "manipulated", what, "la variable manipulada"), f"«manipulated» de {what}",
                                   True)
        for key, purpose in PID_SETTINGS:
            u[key] = float(_need(u, key, what, purpose))
        if u["output_span"] <= 0 or u["output_max"] <= u["output_min"]:
            raise ValueError(f"En {what}: output_span debe ser > 0 y output_max > output_min.")
        action = _need(u, "reverse_acting", what, "el sentido de acción: true si la manipulada debe SUBIR cuando la medida "
                       "supera la consigna (p. ej. nivel con válvula de salida), false si debe bajar")
        if not isinstance(action, bool):
            raise ValueError(f"«reverse_acting» de {what} debe ser true o false.")

    for name, s in S.items():
        _validate_stream(name, s, compounds, U, producer, consumer)

    n_steps = int(round(horizon / dt))
    n_settle = int(round(settle / dt))
    if n_steps < 2:
        raise ValueError("«duration_s» debe cubrir al menos dos pasos de integración («step_s»).")
    if n_steps + n_settle > MAX_STEPS:
        raise ValueError(f"La corrida pide {n_steps + n_settle} pasos; el límite es {MAX_STEPS}. Usa un «step_s» mayor o "
                         "un horizonte más corto.")

    if not monitors:
        raise ValueError("Falta «monitors»: las variables a registrar, cada una {\"object\", \"variable\"}.")
    mons: list[dict] = []
    for m in monitors:
        t = _target(m, "un elemento de «monitors»", False)
        if t["label"] not in [x["label"] for x in mons]:
            mons.append(t)

    evs = []
    for e in events or []:
        what = "un elemento de «events»"
        if not isinstance(e, dict):
            raise ValueError(f"{what} debe ser un diccionario.")
        te = _positive(_need(e, "time_s", what, "el instante del cambio escalón"), "time_s", what, allow_zero=True)
        obj = _need(e, "object", what, "la corriente o equipo que cambia")
        if e.get("mole_fractions") is not None:
            if obj not in S or obj in producer:
                raise ValueError("Un evento de «mole_fractions» solo aplica a una corriente de alimentación (se recibió "
                                 f"«{obj}»).")
            z = _fractions(e["mole_fractions"], compounds, f"el evento en «{obj}»")
            changes = [{"object": obj, "variable": f"mole_fraction:{c}", "property": f"PROP_MS_102/{c}", "units": "",
                        "value": v} for c, v in z.items()]
        else:
            t = _target(e, what, True)
            back_pressure = obj in producer and U[producer[obj]]["kind"] == "valve" and t["variable"] == "pressure_Pa"
            if obj in S and obj in producer and not back_pressure:
                raise ValueError(f"El evento cambia «{t['variable']}» de «{obj}», que la calcula el equipo "
                                 f"«{producer[obj]}»; los eventos de corriente solo aplican a alimentaciones (o a la "
                                 "presión aguas abajo de una válvula).")
            if obj in S and t["variable"] in ("mass_flow_kg_s", "molar_flow_mol_s") and S[obj]["dynamic_spec"] != "flow":
                raise ValueError(f"El evento cambia el caudal de «{obj}», que es dynamic_spec \"pressure\": su caudal lo "
                                 "calcula la válvula.")
            changes = [dict(t, value=float(_need(e, "value", what, "el nuevo valor, en SI")))]
        evs.append({"time_s": te, "changes": changes})
    evs.sort(key=lambda x: x["time_s"])

    return {"compounds": compounds, "property_package": property_package, "streams": S, "units": U, "producer": producer,
            "consumer": consumer, "step_s": dt, "n_steps": n_steps, "n_settle": n_settle, "monitors": mons, "events": evs}


# ----------------------------------------------------------------------------- one DWSIM run
def _run_once(spec: dict, dt: float, n_steps: int, n_settle: int) -> dict:
    """Build the flowsheet, initialise it and integrate `n_settle + n_steps` steps of `dt`. Everything is DWSIM's."""
    net = _net()
    Q, System, Valve = net["Q"], net["System"], net["Valve"]
    if not hasattr(net["PP"], spec["property_package"]):
        raise ValueError(f"Paquete termodinámico «{spec['property_package']}» desconocido en DWSIM.")
    fs = net["Flowsheet"].Create("dynamic")
    for c in spec["compounds"]:
        try:
            fs = fs.WithCompound(c)
        except Exception as ex:  # noqa: BLE001 - DWSIM raises a .NET exception for an unknown compound
            raise ValueError(f"DWSIM no conoce el compuesto «{c}»: {getattr(ex, 'Message', ex)}") from None
    fs = fs.WithPropertyPackage(getattr(net["PP"], spec["property_package"]))

    def composition(z: dict[str, float]):
        pairs = list(z.items())

        def _set(cb):
            for name, x in pairs:
                cb.Mole(name, float(x))

        return System.Action[net["CompositionBuilder"]](_set)

    S, U, producer = spec["streams"], spec["units"], spec["producer"]
    seed = next(s for n, s in S.items() if n not in producer)      # numerical seed for streams DWSIM will compute
    sb: dict[str, Any] = {}
    for name, s in S.items():
        b = fs.AddMaterialStream(name)
        b = b.At(Q.Kelvin(float(s.get("temperature_K") or seed["temperature_K"])),
                 Q.Pascal(float(s.get("pressure_Pa") or seed["pressure_Pa"])))
        if name not in producer:
            if s.get("mass_flow_kg_s") is not None:
                b = b.WithMassFlow(Q.KgPerSecond(s["mass_flow_kg_s"]))
            else:
                b = b.WithMolarFlow(Q.MolPerSecond(s["molar_flow_mol_s"]))
            b = b.WithComposition(composition(s["mole_fractions"]))
        sb[name] = b.AsFlowSpec() if s["dynamic_spec"] == "flow" else b.AsPressureSpec()

    ub: dict[str, Any] = {}
    kinds = Valve.OpeningKvRelationshipType
    relation = {"linear": kinds.Linear, "equal_percentage": kinds.EqualPercentage, "quick_opening": kinds.QuickOpening}
    for name, u in U.items():
        if u["kind"] == "tank":
            ub[name] = (fs.AddTank(name).WithVolume(Q.CubicMeters(u["volume_m3"])).WithHeight(Q.Meters(u["height_m"]))
                        .ConnectFeed(sb[u["feed"]], 0).ConnectProduct(sb[u["product"]], 0))
        elif u["kind"] == "valve":
            # the steady pass only initialises stream states; a shut valve has no steady state, so it is solved open
            steady_opening = u["opening_pct"] if u["opening_pct"] > 0 else 100.0
            ub[name] = (fs.AddValve(name).WithCalcMode(Valve.CalculationMode.Kv_Liquid).WithKv(u["kv"])
                        .WithOpeningKvRelationship(relation[u["characteristic"]], float(u.get("rangeability", 50.0)))
                        .WithOpeningPercent(steady_opening).ConnectFeed(sb[u["feed"]], 0).ConnectProduct(sb[u["product"]], 0))
    for name, u in U.items():
        if u["kind"] == "pid":
            c, m = u["controlled"], u["manipulated"]
            b = (fs.AddPIDController(name).Controls(c["object"], c["property"], c["units"])
                 .Manipulates(m["object"], m["property"], m["units"]).WithSetPoint(u["setpoint"])
                 .WithTuning(u["kp"], u["ki"], u["kd"]).WithOutputLimits(u["output_min"], u["output_max"])
                 .WithOffset(u["output_bias"]).ReverseActing(bool(u["reverse_acting"])))
            b.Object.ManipulatedVariableSpan = u["output_span"]
            ub[name] = b

    fs.AutoLayout()
    steady_errors = [str(getattr(e, "Message", e)) for e in (fs.TrySolve() or [])]
    out: dict[str, Any] = {"steady_errors": steady_errors, "completed": False, "errors": [], "tanks": {}}
    if steady_errors:
        return out

    # initial conditions and boundary pressures (the steady pass overwrote the pressures downstream of each valve)
    for name, s in S.items():
        if name in producer and U[producer[name]]["kind"] == "valve":
            sb[name].WithPressure(Q.Pascal(s["pressure_Pa"]))
    for name, u in U.items():
        if u["kind"] == "valve":
            ub[name].WithOpeningPercent(u["opening_pct"]).WithOpeningSetpoint(u["opening_pct"])
        elif u["kind"] == "tank":
            tank = ub[name].Object
            content = sb[u["feed"]].Object.CloneXML()
            content.SetFlowsheet(fs.Inner)
            content.PropertyPackage = tank.PropertyPackage
            ic = u.get("initial_content")
            if isinstance(ic, dict):
                content.SetTemperature(ic["temperature_K"])
                z = System.Array[System.Double]([ic["mole_fractions"][c] for c in spec["compounds"]])
                content.SetOverallComposition(z)
            content.SetPressure(P_ATM_DWSIM)
            content.PropertyPackage.CurrentMaterialStream = content
            content.Calculate(True, True)
            rho = content.Phases[3].Properties.density
            rho = float(rho) if rho is not None else 0.0
            if u["initial_level_m"] > 0 and not rho > 0:
                raise ValueError(f"El contenido inicial del tanque «{name}» no tiene fase líquida a 101325 Pa: no se puede "
                                 "convertir el nivel inicial en inventario.")
            content.SetMassFlow(u["initial_level_m"] / u["height_m"] * u["volume_m3"] * rho)
            tank.AccumulationStream = content
            out["tanks"][name] = {"initial_liquid_density_kg_m3": rho}

    hidden = []
    for name, u in U.items():
        if u["kind"] == "tank":
            hidden += [(u["feed"], "PROP_MS_2", "kg/s", f"__in:{name}"), (u["product"], "PROP_MS_2", "kg/s", f"__out:{name}")]
    watch = [(m["object"], m["property"], m["units"], m["label"]) for m in spec["monitors"]] + hidden

    def integrate(tag: str, step: float, steps: int, events: list[dict]):
        ib = (fs.Dynamics.DefineIntegrator(tag).WithIntegrationStep(Q.Seconds(step))
              .WithDuration(Q.Seconds((steps - 0.5) * step)))
        for obj, prop, units, label in watch:
            ib = ib.Monitor(obj, prop, units, label)
        sch = fs.Dynamics.DefineSchedule(tag).WithIntegrator(tag).UseCurrentStateAsInitial(True)
        if events:
            es = fs.Dynamics.DefineEventSet(tag)
            for k, ev in enumerate(events):
                for ch in ev["changes"]:
                    es.AddStepChange(ch["object"], ch["property"], float(ch["value"]), Q.Seconds(ev["stamp_s"]),
                                     ch["units"], f"{k}:{ch['object']}.{ch['variable']}")
            sch = sch.WithEventSet(tag)
        sch.MakeCurrent()
        return fs.RunDynamics(tag).WithHistorian(False).FromCurrentState(True).Execute()

    def failure(res) -> list[str]:
        errs = [str(getattr(e, "Message", e)) for e in (res.Errors or [])]
        if not errs and res.Error is not None:
            errs = [str(res.Error.Message)]
        return errs or ["la integración se detuvo antes del horizonte"]

    # priming step: levels, hydrostatic pressures and valve flows consistent with the initial inventories (t = 0)
    prime = integrate("priming", PRIMING_STEP_S, 1, [])
    if not prime.Completed:
        out["errors"] = ["paso de cebado: " + e for e in failure(prime)]
        return out
    first = {label: float(prime.GetSeries(label).Values[0]) for *_, label in watch}
    for name in out["tanks"]:
        out["tanks"][name]["initial_mass_kg"] = float(ub[name].Object.AccumulationStream.GetMassFlow())

    # an event asked at t acts on the steps that start at t or later: DWSIM applies an event stamped in
    # [k dt, (k+1) dt) after step k, and one stamped at 0 before the first step
    events, t0 = [], n_settle * dt
    for ev in spec["events"]:
        k = int(math.ceil(ev["time_s"] / dt - 1e-9))
        stamp = 0.0 if (k + n_settle) == 0 else t0 + (k - 0.5) * dt
        events.append(dict(ev, effective_time_s=k * dt, stamp_s=stamp))
    total = n_settle + n_steps
    res = integrate("run", dt, total, events)
    out["wall_s"] = float(res.WallClock.TotalSeconds)
    out["steps_solved"] = int(res.Steps)
    if not res.Completed:
        out["errors"] = failure(res)
        return out
    series = {label: [first[label]] + [float(v) for v in res.GetSeries(label).Values] for *_, label in watch}
    if any(len(v) != total + 1 for v in series.values()):
        out["errors"] = ["DWSIM devolvió un número de muestras distinto del número de pasos pedido"]
        return out
    for name, u in U.items():
        if u["kind"] != "tank":
            continue
        w_in, w_out = series.pop(f"__in:{name}"), series.pop(f"__out:{name}")
        m_end = float(ub[name].Object.AccumulationStream.GetMassFlow())
        t = out["tanks"][name]
        fed, drawn = dt * math.fsum(w_in[1:]), dt * math.fsum(w_out[:-1])     # the explicit rule DWSIM integrates with
        trap = dt * (math.fsum(w_in) - 0.5 * (w_in[0] + w_in[-1]) - math.fsum(w_out) + 0.5 * (w_out[0] + w_out[-1]))
        scale = max(fed, drawn, t["initial_mass_kg"], m_end, 1e-12)
        t.update(final_mass_kg=m_end, fed_kg=fed, drawn_kg=drawn,
                 mass_balance_rel_residual=abs((m_end - t["initial_mass_kg"]) - (fed - drawn)) / scale,
                 mass_balance_trapezoid_rel_residual=abs((m_end - t["initial_mass_kg"]) - trap) / scale,
                 emptied=bool(m_end <= 0.0 and (t["initial_mass_kg"] > 0 or fed > 0)))
    out.update(completed=True, series=series, events=[{k: v for k, v in e.items() if k != "stamp_s"} for e in events])
    return out


# ----------------------------------------------------------------------------- public engine
def run_dynamic(compounds: list[str], property_package: str, streams: list[dict], units: list[dict], step_s: float,
                duration_s: float, monitors: list[dict], events: list[dict] | None = None, settle_s: float | None = None,
                check_step_halving: bool = True, step_tolerance: float = STEP_TOL_DEFAULT,
                settle_tolerance: float = SETTLE_TOL_DEFAULT, max_points: int = 400) -> dict:
    """Integrate a flowsheet of open tanks, liquid Kv valves and PID controllers with DWSIM's dynamic mode.

    streams: {name, dynamic_spec "flow"|"pressure", temperature_K, pressure_Pa, mass_flow_kg_s|molar_flow_mol_s,
    mole_fractions}; units: tank {name, feed, product, volume_m3, height_m, initial_level_m, initial_content}, valve
    {name, feed, product, kv, opening_pct, characteristic, rangeability}, pid {name, controlled, manipulated, setpoint,
    kp, ki, kd, reverse_acting, output_bias, output_span, output_min, output_max}; monitors and events address
    {object, variable}. settle_s integrates that long with no events before t = 0 (steady-state initialisation in
    dynamic mode) and checks it settled. Returns the time series (SI), the tank inventories and the evidence of the run."""
    spec = _validate(compounds, property_package, streams, units, step_s, duration_s, monitors, events, settle_s)
    dt, n, ns = spec["step_s"], spec["n_steps"], spec["n_settle"]
    warnings: list[str] = []
    reasons: list[str] = []
    if abs(n * dt - float(duration_s)) > 1e-9 * max(1.0, float(duration_s)):
        warnings.append(f"El horizonte se redondeó a {n} pasos de {dt:g} s = {n * dt:g} s.")

    run = _run_once(spec, dt, n, ns)
    result: dict[str, Any] = {
        "method": ("Modo dinámico de DWSIM (IntegratorRunner vía Fluent API y pythonnet): Euler explícito "
                   f"secuencial-modular de paso fijo {dt:g} s; arranque con una pasada estacionaria de DWSIM, condiciones "
                   f"iniciales impuestas y un paso de cebado de {PRIMING_STEP_S:g} s; ningún cálculo físico se hace fuera de "
                   "DWSIM"),
        "step_s": dt, "duration_s": n * dt, "settle_s": ns * dt, "warnings": warnings, "not_included": list(NOT_INCLUDED),
        "assumptions": _assumptions(spec),
    }
    evidence: dict[str, Any] = {"verified": False, "reasons": reasons, "steady_state_solved": not run["steady_errors"],
                                "integration_completed": bool(run["completed"]), "steps_solved": run.get("steps_solved", 0)}
    result["evidence"] = evidence
    if run["steady_errors"]:
        reasons.append("La pasada estacionaria de inicialización de DWSIM no resolvió: "
                       + "; ".join(run["steady_errors"][:3]))
        if any(s["dynamic_spec"] == "pressure" and name not in spec["producer"] for name, s in spec["streams"].items()):
            reasons.append("Hay alimentaciones con dynamic_spec \"pressure\": su caudal de arranque debe ser compatible con "
                           "el Kv y las presiones dadas (la pasada estacionaria calcula la caída de presión de la válvula "
                           "con ese caudal).")
        return _clean(result)
    if not run["completed"]:
        reasons.append("DWSIM no terminó la integración: " + "; ".join(run["errors"][:3]))
        return _clean(result)

    full_t = [(k - ns) * dt for k in range(ns + n + 1)]
    series = run["series"]
    labels = [m["label"] for m in spec["monitors"]]

    # evidence 1: mass balance of every tank with DWSIM's explicit rule; tanks that ran empty or over their height
    evidence["mass_balance_max_rel_residual"] = max((t["mass_balance_rel_residual"] for t in run["tanks"].values()),
                                                    default=0.0)
    evidence["mass_balance_tolerance"] = MASS_BALANCE_TOL
    for name, t in run["tanks"].items():
        height = spec["units"][name]["height_m"]
        if t["emptied"]:
            warnings.append(f"El tanque «{name}» se vació: DWSIM recorta el inventario a cero (el último paso extrae más de "
                            "lo que queda) y el nivel publicado se queda en su último valor positivo en vez de cero.")
        if t["mass_balance_rel_residual"] > MASS_BALANCE_TOL:
            reasons.append(f"El balance de materia del tanque «{name}» no cierra (residuo relativo "
                           f"{t['mass_balance_rel_residual']:.2e})" + (" porque se vació." if t["emptied"] else "."))
        level = series.get(f"{name}.level_m")
        if level is not None and max(level) > height:
            text = (f"El nivel del tanque «{name}» superó su altura ({max(level):.4g} m > {height:g} m): el tanque de DWSIM "
                    "no tiene rebose y el nivel sigue subiendo; el resultado deja de ser físico desde ahí.")
            warnings.append(text)
            reasons.append(text)

    # evidence 2: steady-state initialisation settled
    if ns:
        k0 = int(0.8 * ns)
        drift = {lab: (max(series[lab][k0:ns + 1]) - min(series[lab][k0:ns + 1])) / max(abs(series[lab][ns]), 1e-9)
                 for lab in labels}
        evidence["settle_max_rel_drift"] = max(drift.values())
        evidence["settle_tolerance"] = settle_tolerance
        if evidence["settle_max_rel_drift"] > settle_tolerance:
            lab = max(drift, key=drift.get)
            reasons.append(f"El estado no se asentó en settle_s = {ns * dt:g} s: «{lab}» aún cambia {drift[lab]:.2e} "
                           "relativo en el último quinto de la ventana; t = 0 no es un estado estacionario.")

    # evidence 3: step independence, the same case at step / 2 (time-averaged difference; the pointwise maximum is
    # informative: a jump that DWSIM applies one step late never converges pointwise)
    if check_step_halving:
        half = _run_once(spec, dt / 2.0, 2 * n, 2 * ns)
        if not half["completed"]:
            reasons.append("La corrida de comprobación con el paso a la mitad no terminó: "
                           + "; ".join((half["steady_errors"] + half["errors"])[:3]))
        else:
            per_mean, per_max = {}, {}
            for lab in labels:
                y, yh = series[lab][ns:], half["series"][lab][2 * ns::2]
                span = max(max(yh) - min(yh), SPAN_FLOOR_REL * max(abs(v) for v in yh), 1e-12)
                diff = [abs(a - b) for a, b in zip(y, yh)]
                per_mean[lab], per_max[lab] = math.fsum(diff) / len(diff) / span, max(diff) / span
            evidence["step_halving_mean_rel_diff"] = per_mean
            evidence["step_halving_pointwise_max_rel_diff"] = per_max
            evidence["step_halving_worst_mean_rel_diff"] = max(per_mean.values())
            evidence["step_tolerance"] = step_tolerance
            if evidence["step_halving_worst_mean_rel_diff"] > step_tolerance:
                lab = max(per_mean, key=per_mean.get)
                reasons.append(f"El resultado depende del paso: «{lab}» cambia en promedio {per_mean[lab]:.2e} de su rango "
                               f"al usar step_s / 2 (tolerancia {step_tolerance:g}). Repite con un paso menor.")
    else:
        reasons.append("No se comprobó la independencia del paso (check_step_halving = false).")

    evidence["verified"] = not reasons
    stride = max(1, int(math.ceil((n + 1) / max(int(max_points), 2))))
    keep = sorted(set(range(ns, ns + n + 1, stride)) | {ns + n})
    applied = [{"time_s": e["time_s"], "effective_time_s": e["effective_time_s"],
                "changes": [{k: c[k] for k in ("object", "variable", "value")} for c in e["changes"]]}
               for e in run["events"]]
    result.update(
        time_s=[full_t[k] for k in keep], series={lab: [series[lab][k] for k in keep] for lab in labels},
        units={m["label"]: m["units"] for m in spec["monitors"]},
        initial={lab: series[lab][ns] for lab in labels}, final={lab: series[lab][-1] for lab in labels},
        minimum={lab: min(series[lab][ns:]) for lab in labels}, maximum={lab: max(series[lab][ns:]) for lab in labels},
        events_applied=applied, tanks=run["tanks"], samples_returned=len(keep), samples_computed=n + 1,
        wall_clock_s=run.get("wall_s"))
    for e in applied:
        if e["effective_time_s"] >= n * dt:
            warnings.append(f"El evento de t = {e['time_s']:g} s cae fuera del horizonte y no actuó.")
        elif abs(e["effective_time_s"] - e["time_s"]) > 1e-9:
            warnings.append(f"El evento de t = {e['time_s']:g} s actúa desde t = {e['effective_time_s']:g} s (siguiente "
                            "inicio de paso).")
    return _clean(result)


def _assumptions(spec: dict) -> list[str]:
    out = [
        "t = 0 es la condición inicial dada; DWSIM etiqueta como t = 0 la muestra posterior al primer paso, aquí cada "
        "muestra se informa en el tiempo físico al que corresponde (n · paso)",
        "Un evento pedido en t actúa sobre los pasos que empiezan en t o después; la muestra en t todavía muestra el valor "
        "anterior",
        f"Tanque abierto de DWSIM: espacio de vapor fijo a {P_ATM_DWSIM:g} Pa, mezcla perfecta, sin pérdidas de calor; "
        f"presión de salida = {P_ATM_DWSIM:g} + ρ·{G_DWSIM}·nivel (DWSIM usa g = {G_DWSIM} m/s²)",
        "Válvula de líquido de DWSIM: W[kg/h] = Kv(apertura)·√(ρ[kg/m³]·999.1·ΔP[bar]), isoentálpica, con flujo "
        "estrangulado en ΔP = FL²·(P1 − FF·Pv) y FL = 0.9 (valor por defecto de DWSIM)",
        "La pasada estacionaria previa solo inicializa el estado termodinámico de las corrientes; presiones de contorno, "
        "niveles, contenidos y aperturas se imponen después con los datos dados",
    ]
    if any(u["kind"] == "valve" and u["opening_pct"] == 0 for u in spec["units"].values()):
        out.append("Las válvulas que arrancan cerradas se resuelven abiertas al 100 % en la pasada estacionaria (una "
                   "válvula cerrada no tiene estado estacionario) y se cierran antes de integrar")
    if any(u["kind"] == "pid" for u in spec["units"].values()):
        out.append("PID de DWSIM (forma de posición): error e = (medida − consigna)/|consigna inicial|; salida = Kp·e + "
                   "Ki·∫e dt + Kd·de/dt; manipulada = output_bias ± salida·output_span (+ si reverse_acting), recortada a "
                   "[output_min, output_max] con anti-windup; el controlador actúa una vez por paso, después de resolver "
                   "el proceso, y ve un cambio de consigna un paso después del evento")
    return out


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x
