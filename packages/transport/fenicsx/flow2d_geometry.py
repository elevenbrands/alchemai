"""Geometry, boundary selectors, safe expressions and gmsh meshing for `flow2d_solver.py` (runs in the FEniCSx env).

Everything here works in the NON-DIMENSIONAL coordinates of the solver (lengths / L_ref). No dolfinx import except in
`build_mesh` (gmsh → dolfinx mesh). Coordinates: planar (x, y); axisymmetric (x = z axial, y = r radial, axis y = 0).
"""
from __future__ import annotations

import ast
import math

import numpy as np

EPS = 1e-7

# ---------------------------------------------------------------- safe expressions (no eval/exec)
_FUNCS = {"sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "tanh": np.tanh,
          "log10": np.log10, "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan, "sinh": np.sinh, "cosh": np.cosh}
_BIN = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power}
_CONST = {"pi": math.pi}


def parse_expr(text, names):
    """Parse an arithmetic expression of the given names into a checked AST (numbers, + - * / **, sign, 1-arg functions)."""
    if isinstance(text, (int, float)):
        return ast.Constant(float(text))
    if not isinstance(text, str) or not text.strip() or len(text) > 400:
        raise ValueError(f"expresión inválida: {text!r}")
    tree = ast.parse(text.replace("^", "**"), mode="eval").body

    def chk(n, depth=0):
        if depth > 40:
            raise ValueError("expresión demasiado anidada")
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return
        if isinstance(n, ast.Name) and (n.id in names or n.id in _CONST):
            return
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            chk(n.left, depth + 1), chk(n.right, depth + 1)
            return
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            chk(n.operand, depth + 1)
            return
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and len(n.args) == 1 and not n.keywords:
            chk(n.args[0], depth + 1)
            return
        raise ValueError(f"elemento no permitido en la expresión {text!r}: {type(n).__name__}")
    chk(tree)
    return tree


def eval_expr(n, env):
    if isinstance(n, ast.Constant):
        return float(n.value)
    if isinstance(n, ast.Name):
        return env[n.id] if n.id in env else _CONST[n.id]
    if isinstance(n, ast.BinOp):
        return _BIN[type(n.op)](eval_expr(n.left, env), eval_expr(n.right, env))
    if isinstance(n, ast.UnaryOp):
        v = eval_expr(n.operand, env)
        return -v if isinstance(n.op, ast.USub) else v
    return _FUNCS[n.func.id](eval_expr(n.args[0], env))


# ---------------------------------------------------------------- geometry
class Geometry:
    def __init__(self, mode, rectangles, obstacles):
        self.axi = mode == "axisymmetric"
        self.rects = [tuple(float(v) for v in r) for r in rectangles]
        self.obstacles = obstacles
        self.bbox = (min(r[0] for r in self.rects), min(r[1] for r in self.rects),
                     max(r[2] for r in self.rects), max(r[3] for r in self.rects))

    def in_obstacle(self, o, x, y, tol=0.0):
        if o["type"] == "circle":
            return (x - o["center"][0]) ** 2 + (y - o["center"][1]) ** 2 <= (o["radius"] + tol) ** 2
        b = o["box"]
        return (x >= b[0] - tol) & (x <= b[2] + tol) & (y >= b[1] - tol) & (y <= b[3] + tol)

    def in_fluid(self, x, y):
        """Open-set membership (vectorised): strictly inside some rectangle and outside every obstacle."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        ins = np.zeros(np.broadcast(x, y).shape, bool)
        for r in self.rects:
            ins |= (x > r[0] - EPS) & (x < r[2] + EPS) & (y > r[1] - EPS) & (y < r[3] + EPS)
        for o in self.obstacles:
            ins &= ~self.in_obstacle(o, x, y, tol=-EPS)
        return ins

    def obstacle_box(self, o):
        if o["type"] == "circle":
            c, r = o["center"], o["radius"]
            return (c[0] - r, c[1] - r, c[0] + r, c[1] + r)
        return tuple(o["box"])

    def obstacle_size(self, o, stream_axis):
        """Transverse extent d (planar) or 2·r_max (axisymmetric) and frontal area per unit depth / π r_max²."""
        b = self.obstacle_box(o)
        if self.axi:
            rmax, rmin = min(b[3], self.bbox[3]), max(b[1], 0.0)
            return 2 * rmax, math.pi * (rmax ** 2 - rmin ** 2)
        d = (b[3] - b[1]) if stream_axis == 0 else (b[2] - b[0])
        return d, d


def on_line(b, x, y, tol=1e-6):
    """Vectorised: points on the boundary line selector b = {axis, value, range}."""
    a = b["axis"]
    c, t = (x, y) if a == "x" else (y, x)
    lo, hi = b["range"]
    return (np.abs(c - b["value"]) < tol) & (t > lo - tol) & (t < hi + tol)


# ---------------------------------------------------------------- inlet profiles
def annulus_shape(kappa):
    """u/ū for fully developed flow in an annulus r ∈ [κb, b] (BSL eq. 2.4-14/16), as a function of ξ = r/b."""
    c = (1 - kappa ** 2) / math.log(1 / kappa)
    mean = 1 + kappa ** 2 - c
    return lambda xi: 2 * (1 - xi ** 2 + c * np.log(np.maximum(xi, 1e-300))) / mean


def profile_shape(geo: Geometry, b, sym_lo, sym_hi, profile):
    """Dimensionless shape f(t) with unit (area-weighted) mean on the line selector b; t is the tangent coordinate."""
    lo, hi = b["range"]
    w = hi - lo
    if profile == "uniform" or (sym_lo and sym_hi):
        return lambda t: np.ones_like(t)
    if geo.axi and b["axis"] == "x":
        if sym_lo:                                                    # axis → wall: Hagen-Poiseuille
            return lambda t: 2 * (1 - (t / hi) ** 2)
        f = annulus_shape(lo / hi)
        return lambda t: f(t / hi)
    if sym_lo:
        return lambda t: 1.5 * (1 - ((t - lo) / w) ** 2)
    if sym_hi:
        return lambda t: 1.5 * (1 - ((hi - t) / w) ** 2)
    return lambda t: 1.5 * (1 - ((2 * t - lo - hi) / w) ** 2)


def developed_gradient(geo: Geometry, axis, lo, hi, sym_lo, sym_hi, nu, umean):
    """Exact fully developed −dp/ds (kinematic, ρ = 1) and u_max/ū for a straight section; None when not applicable."""
    w = hi - lo
    if sym_lo and sym_hi:
        return None
    if geo.axi:
        if axis != "x":
            return None
        if sym_lo:
            return 8 * nu * umean / hi ** 2, 2.0                      # Hagen-Poiseuille
        k = lo / hi
        c = (1 - k ** 2) / math.log(1 / k)
        f = annulus_shape(k)
        xi = np.linspace(k, 1, 2001)
        return 8 * nu * umean / (hi ** 2 * (1 + k ** 2 - c)), float(np.max(f(xi)))
    if sym_lo or sym_hi:
        return 3 * nu * umean / w ** 2, 1.5                           # half channel of height w
    return 12 * nu * umean / w ** 2, 1.5                              # plane Poiseuille


# ---------------------------------------------------------------- mesh
def build_mesh(geo: Geometry, boundaries, mp, factor=1.0):
    """gmsh (OCC) → dolfinx mesh. `factor` divides every target size (mesh-refinement evidence). Returns (mesh, info)."""
    import gmsh
    from dolfinx.io import gmsh as gio
    from mpi4py import MPI

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        surf = [(2, occ.addRectangle(r[0], r[1], 0, r[2] - r[0], r[3] - r[1])) for r in geo.rects]
        if len(surf) > 1:
            surf, _ = occ.fuse(surf[:1], surf[1:])
        tools = []
        for o in geo.obstacles:
            if o["type"] == "circle":
                tools.append((2, occ.addDisk(o["center"][0], o["center"][1], 0, o["radius"], o["radius"])))
            else:
                b = o["box"]
                tools.append((2, occ.addRectangle(b[0], b[1], 0, b[2] - b[0], b[3] - b[1])))
        if tools:
            surf, _ = occ.cut(surf, tools)
        pts = []
        for b in boundaries:
            if b.get("axis"):
                for t in b["range"]:
                    xy = (b["value"], t) if b["axis"] == "x" else (t, b["value"])
                    e = 1e-5
                    near = geo.in_fluid(np.array([xy[0] + dx for dx in (-e, 0, e) for _ in range(3)]),
                                        np.array([xy[1] + dy for _ in range(3) for dy in (-e, 0, e)]))
                    if near.any() and not near.all():                  # only points ON the fluid boundary
                        pts.append((0, occ.addPoint(xy[0], xy[1], 0)))
        if pts:
            occ.fragment(surf, pts)
        occ.synchronize()
        surfs = gmsh.model.getEntities(2)
        if not surfs:
            raise ValueError("la geometría no deja dominio de fluido (¿obstáculo mayor que el canal?)")
        hf, g = mp["size"] / factor, mp["growth"]
        fields = []
        fld = gmsh.model.mesh.field
        curves = [t for _, t in gmsh.model.getBoundary(surfs, oriented=False, combined=True)]
        for o in geo.obstacles:
            mine = []
            for t in curves:
                lo, hi = gmsh.model.getParametrizationBounds(1, t)
                p = gmsh.model.getValue(1, t, [(lo[0] + hi[0]) / 2])
                if geo.in_obstacle(o, p[0], p[1], tol=1e-6):
                    mine.append(t)
            if not mine:
                continue
            d = fld.add("Distance")
            fld.setNumbers(d, "CurvesList", mine)
            fld.setNumber(d, "Sampling", 200)
            m = fld.add("MathEval")
            fld.setString(m, "F", f"Min({hf!r}, {o['mesh_size'] / factor!r} + {g!r}*F{d})")
            fields.append(m)
        for bx in mp.get("boxes", []):
            f = fld.add("Box")
            b = bx["box"]
            fld.setNumber(f, "VIn", bx["size"] / factor)
            fld.setNumber(f, "VOut", hf)
            for k, v in zip(("XMin", "YMin", "XMax", "YMax"), b):
                fld.setNumber(f, k, v)
            fld.setNumber(f, "Thickness", max(hf - bx["size"] / factor, 0.0) / max(g, 1e-3))
            fields.append(f)
        if fields:
            mn = fld.add("Min")
            fld.setNumbers(mn, "FieldsList", fields)
            fld.setAsBackgroundMesh(mn)
            for k in ("MeshSizeExtendFromBoundary", "MeshSizeFromPoints", "MeshSizeFromCurvature"):
                gmsh.option.setNumber(f"Mesh.{k}", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", hf)
        if not fields:
            gmsh.option.setNumber("Mesh.MeshSizeMin", hf * 0.999)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.addPhysicalGroup(2, [t for _, t in surfs], 1)
        gmsh.model.mesh.generate(2)
        order = 2 if any(o["type"] == "circle" for o in geo.obstacles) else 1
        if order == 2:
            gmsh.model.mesh.setOrder(2)
        md = gio.model_to_mesh(gmsh.model, MPI.COMM_SELF, 0, gdim=2)
        return md.mesh, {"geometry_order": order, "n_surfaces": len(surfs)}
    finally:
        gmsh.finalize()
