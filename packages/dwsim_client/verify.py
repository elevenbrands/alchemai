"""Deterministic verification of a solved DWSIM flowsheet.

The agent never reports a number that did not pass through here. For every unit operation:
  * convergence (`Calculated`, `ErrorMessage`, solver exceptions),
  * total mass balance (kg/s),
  * per-compound molar balance (mol/s) — skipped for reactors, where only mass is conserved,
  * energy balance: sum(H_out) - sum(H_in) against attached energy streams, or, when none is
    attached, against the unit's own duty (`DeltaQ`, sign-agnostic).

Everything is read back from DWSIM's object model (`Flowsheet.Inner.SimulationObjects`), not
from the builders, so it works on flowsheets built by anyone (agent, file, MCP).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

# DWSIM's flash and shortcut solvers close balances to ~1e-5 relative; 1e-4 (0.01 %) is the
# engineering-grade "closed" threshold. Compressor/heater steps close to ~1e-10.
REL_TOL = 1e-4      # relative tolerance for mass and compound balances
ENERGY_TOL = 1e-4   # relative tolerance for energy balance (kW scale)


@dataclass
class StreamView:
    name: str
    mass_kg_s: float
    molar_mol_s: float
    enthalpy_kw: float             # mass enthalpy (kJ/kg) × mass flow (kg/s) = kW
    compounds_mol_s: dict[str, float]
    temperature_k: float
    pressure_pa: float
    vapor_fraction: float


@dataclass
class UnitCheck:
    name: str
    kind: str
    calculated: bool
    error: str
    inlets: list[str]
    outlets: list[str]
    energy_in_kw: float
    energy_out_kw: float
    duty_kw: float | None
    mass_in: float
    mass_out: float
    mass_rel_err: float
    compound_max_rel_err: float | None
    enthalpy_in_kw: float
    enthalpy_out_kw: float
    energy_residual_kw: float
    energy_rel_err: float
    ok: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class Report:
    solved: bool
    solver_errors: list[str]
    units: list[UnitCheck]
    streams: dict[str, StreamView]

    @property
    def ok(self) -> bool:
        return self.solved and all(u.ok for u in self.units)

    def to_dict(self) -> dict:
        return {
            "solved": self.solved,
            "ok": self.ok,
            "solver_errors": self.solver_errors,
            "units": [asdict(u) for u in self.units],
            "streams": {k: asdict(v) for k, v in self.streams.items()},
        }

    def markdown(self) -> str:
        lines = [f"**Solved:** {self.solved} · **All checks:** {'✅' if self.ok else '❌'}"]
        if self.solver_errors:
            lines += ["", "Solver errors:"] + [f"- {e}" for e in self.solver_errors]
        lines += ["", "| Unit | Type | Calc | Mass rel. err | Compound rel. err | ΔH − Q (kW) | Energy rel. err | OK |",
                  "|---|---|---|---|---|---|---|---|"]
        for u in self.units:
            cmp_ = "—" if u.compound_max_rel_err is None else f"{u.compound_max_rel_err:.1e}"
            lines.append(f"| {u.name} | {u.kind} | {'✅' if u.calculated else '❌'} | {u.mass_rel_err:.1e} | {cmp_} | "
                         f"{u.energy_residual_kw:+.4g} | {u.energy_rel_err:.1e} | {'✅' if u.ok else '❌'} |")
        notes = [f"- {u.name}: {n}" for u in self.units for n in u.notes]
        if notes:
            lines += ["", "Notes:"] + notes
        return "\n".join(lines)


# ----------------------------------------------------------------------------- object model
def _obj_type(sim_obj) -> str:
    return str(sim_obj.GraphicObject.ObjectType)


def _is_material(sim_obj) -> bool:
    return _obj_type(sim_obj) == "MaterialStream"


def _is_energy(sim_obj) -> bool:
    return _obj_type(sim_obj) == "EnergyStream"


def _prop(obj, name: str, default=None):
    """Read a property by reflection: the dictionary hands out interface wrappers, and
    pythonnet only exposes interface members on them."""
    p = obj.GetType().GetProperty(name)
    if p is None:
        return default
    v = p.GetValue(obj)
    return default if v is None else v


def _stream_view(ms) -> StreamView:
    mass = float(ms.GetMassFlow() or 0.0)
    h_mass = float(ms.GetMassEnthalpy() or 0.0)   # kJ/kg
    compounds = {c.Key: float(c.Value.MolarFlow or 0.0) for c in ms.Phases[0].Compounds}
    props = ms.Phases[0].Properties
    return StreamView(
        name=ms.GraphicObject.Tag,
        mass_kg_s=mass,
        molar_mol_s=float(ms.GetMolarFlow() or 0.0),
        enthalpy_kw=h_mass * mass,
        compounds_mol_s=compounds,
        temperature_k=float(props.temperature or 0.0),
        pressure_pa=float(props.pressure or 0.0),
        vapor_fraction=float(ms.Phases[2].Properties.molarfraction or 0.0),
    )


def _energy_kw(es) -> float:
    v = _prop(es, "EnergyFlow")
    return float(v) if v is not None else 0.0


def _attached(connectors, objects):
    """Yield (sim_obj, direction) for attached connectors; direction is 'from' or 'to'."""
    for c in connectors:
        if not c.IsAttached:
            continue
        ac = c.AttachedConnector
        yield objects.get(ac.AttachedFrom.Name), objects.get(ac.AttachedTo.Name)


def _rel(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / scale


# ----------------------------------------------------------------------------- public API
def verify(fs, rel_tol: float = REL_TOL, energy_tol: float = ENERGY_TOL, solve: bool = True) -> Report:
    """Solve (optionally) and verify a Fluent-API `Flowsheet`. Returns a `Report`."""
    solver_errors: list[str] = []
    solved = True
    if solve:
        errs = fs.TrySolve()
        solver_errors = [str(e).splitlines()[0] for e in errs] if errs is not None else []
        solved = not solver_errors

    inner = fs.Inner
    objects = {kv.Key: kv.Value for kv in inner.SimulationObjects}
    # Concrete MaterialStream objects come from the Fluent accessor; the dictionary only yields
    # ISimulationObject wrappers.
    streams = {o.GraphicObject.Tag: _stream_view(fs.MaterialStream(o.GraphicObject.Tag).Object)
               for o in objects.values() if _is_material(o)}
    units: list[UnitCheck] = []

    for o in objects.values():
        if _is_material(o) or _is_energy(o):
            continue
        go = o.GraphicObject
        me = go.Name
        in_mat, out_mat, e_in, e_out = [], [], 0.0, 0.0
        for src, dst in _attached(go.InputConnectors, objects):
            other = src if (dst is not None and dst.GraphicObject.Name == me) else dst
            if other is None:
                continue
            if _is_material(other):
                in_mat.append(other)
            elif _is_energy(other):
                e_in += _energy_kw(other)
        for src, dst in _attached(go.OutputConnectors, objects):
            other = dst if (src is not None and src.GraphicObject.Name == me) else src
            if other is None:
                continue
            if _is_material(other):
                out_mat.append(other)
            elif _is_energy(other):
                e_out += _energy_kw(other)
        ec = getattr(go, "EnergyConnector", None)
        if ec is not None and ec.IsAttached:
            other = objects.get(ec.AttachedConnector.AttachedTo.Name)
            if other is None or other.GraphicObject.Name == me:
                other = objects.get(ec.AttachedConnector.AttachedFrom.Name)
            if other is not None and _is_energy(other):
                e_out += _energy_kw(other)   # DWSIM draws the duty leaving the unit here

        vin = [streams[m.GraphicObject.Tag] for m in in_mat]
        vout = [streams[m.GraphicObject.Tag] for m in out_mat]
        mass_in, mass_out = sum(v.mass_kg_s for v in vin), sum(v.mass_kg_s for v in vout)
        h_in, h_out = sum(v.enthalpy_kw for v in vin), sum(v.enthalpy_kw for v in vout)
        kind = _obj_type(o)
        notes: list[str] = []

        mass_err = _rel(mass_in, mass_out) if (mass_in or mass_out) else 0.0

        comp_err: float | None = None
        if "Reactor" not in kind and vin and vout:
            names = set().union(*(v.compounds_mol_s for v in vin + vout))
            comp_err = max(
                _rel(sum(v.compounds_mol_s.get(n, 0.0) for v in vin),
                     sum(v.compounds_mol_s.get(n, 0.0) for v in vout))
                if (sum(v.compounds_mol_s.get(n, 0.0) for v in vin) > 1e-9 * max(mass_in, 1e-12)) else 0.0
                for n in names
            )
        elif "Reactor" in kind:
            notes.append("reactor: per-compound balance skipped (only mass conserved)")

        duty = None
        dq = _prop(o, "DeltaQ")
        if dq is not None:
            try:
                duty = float(dq)
            except TypeError:
                duty = None

        delta_h = h_out - h_in
        if e_in or e_out:
            residual = delta_h - (e_in - e_out)
        elif duty is not None and abs(duty) > 1e-9:
            # No energy stream attached: the unit's own duty must explain ΔH, whatever its sign.
            residual = min(abs(delta_h - duty), abs(delta_h + duty)) * (1 if delta_h >= 0 else -1)
            notes.append(f"energy checked against DeltaQ = {duty:.6g} kW (no energy stream attached)")
        else:
            residual = delta_h
        energy_err = abs(residual) / max(abs(h_in), abs(h_out), abs(e_in), abs(e_out), 1e-9)

        calculated = bool(getattr(o, "Calculated", True))
        error = str(getattr(o, "ErrorMessage", "") or "")
        ok = calculated and not error and mass_err <= rel_tol and energy_err <= energy_tol \
            and (comp_err is None or comp_err <= rel_tol)
        units.append(UnitCheck(
            name=go.Tag, kind=kind, calculated=calculated, error=error,
            inlets=[v.name for v in vin], outlets=[v.name for v in vout],
            energy_in_kw=e_in, energy_out_kw=e_out, duty_kw=duty,
            mass_in=mass_in, mass_out=mass_out, mass_rel_err=mass_err,
            compound_max_rel_err=comp_err, enthalpy_in_kw=h_in, enthalpy_out_kw=h_out,
            energy_residual_kw=residual, energy_rel_err=energy_err, ok=ok, notes=notes,
        ))

    return Report(solved=solved, solver_errors=solver_errors, units=units, streams=streams)
