"""Agent-facing wrapper of the laminar 2-D flow tool (`packages.dwsim_client.tools_flow2d`)."""
from __future__ import annotations

import json

from anthropic import beta_tool

from packages.dwsim_client import tools_flow2d as TF


@beta_tool
def laminar_flow_2d(geometry: dict, boundaries: list[dict], density_kg_m3: float | None = None,
                    viscosity_Pa_s: float | None = None, reynolds: float | None = None,
                    reference_length_m: float | None = None, reference_velocity_m_s: float | None = None,
                    stream_direction: str = "+x", points: list[list[float]] | None = None,
                    sections: list[dict] | None = None, pressure_drop_between: list[list[str]] | None = None,
                    scalar: dict | None = None, transient: dict | None = None, mesh: dict | None = None,
                    figure_path: str | None = None, timeout_s: int | None = None) -> str:
    """Laminar incompressible flow on a 2-D geometry YOU describe (steady Navier-Stokes, FEniCSx Taylor-Hood P2-P1 in a
    separate process): channel/tube entry flow, T/L junctions, sudden expansion/contraction, orifice plate, cylinder or
    sphere in a stream. Optional vortex-shedding transient and passive scalar (Nusselt/Sherwood). Lengths in m.

    geometry: {"mode": "planar"|"axisymmetric", "rectangles": [[x0,y0,x1,y1],...] (fluid = their union), "obstacles":
      [{"name","type":"circle","center":[x,y],"radius"} | {"name","type":"rectangle","box":[x0,y0,x1,y1]}]}.
      Axisymmetric: x = axis z, y = radius r >= 0; y = 0 is the axis (automatic). A circle centred on the axis is a
      sphere; a rectangle obstacle hanging from the wall is an orifice plate.
    boundaries: [{"name","type","on",...}]; every boundary not selected is a no-slip wall. "on": {"side": "left"|"right"|
      "bottom"|"top"} of the bounding box (optional "range": [lo,hi]) | {"x": c, "y": [lo,hi]} | {"y": c, "x": [lo,hi]} |
      {"obstacle": name}. Types: "inlet" ("mean_velocity" m/s + "profile" "parabolic"|"uniform", or "velocity": [vx,vy]
      numbers or expressions of x,y such as "1.5*(1-(y/0.01)**2)"); "outlet" (p = "pressure", default 0, normal flow;
      "normal_flow": false = free outflow, use it behind wakes); "symmetry" (slip); "stream" (far field, "velocity":
      [vx,vy]); "wall" (optional "velocity" = moving wall).
    Fluid: density_kg_m3 + viscosity_Pa_s, or reynolds = rho*U_ref*L_ref/mu (then rho = 1: use the dimensionless outputs).
      reference_length_m defaults to the first immersed obstacle's diameter, else the first inlet's width (tube: diameter);
      reference_velocity_m_s to the largest inlet/stream velocity. stream_direction: "+x" (drag / reversed-flow axis).
    points: [[x,y],...] velocity and pressure there. sections: [{"name","x": c,"y": [lo,hi]}] internal lines (flow, mean p).
      pressure_drop_between: [[a,b],...] boundary names (default: every inlet -> outlet pair).
    scalar: {"peclet" (U_ref*L_ref/D) | "diffusivity_m2_s", "values": {boundary or obstacle name: value}} -> steady
      convection-diffusion with the computed velocity; mean Nu/Sh on each non-inlet surface (zero flux elsewhere).
    transient: {"dt_s","t_end_s","perturbation","min_cycles","mesh": "base"|"fine"} -> semi-implicit BDF2 from the steady
      solution (defaults dt = 0.05 L_ref/U_ref, t_end = 150 L_ref/U_ref): state "steady"|"periodic"|"not_settled",
      strouhal, drag_coefficient_mean, lift amplitude. Slow (minutes); a run cut by the time limit says "not_settled".
    mesh: {"size", "obstacle_size", "growth", "boxes": [{"box": [x0,y0,x1,y1], "size"}], "refine_ratio" (1.5),
      "refinement_check" (true), "tolerance" (0.02)}; give "boxes" in wakes and near re-entrant corners.
    timeout_s: default 300 (1800 with transient); the solver process is killed at the limit.

    Returns Re, reference, boundaries {name: flow_out (per unit depth m2/s, or m3/s axisymmetric), mean_pressure},
    mass_balance, pressure_drops, points, sections, straight_sections (fully developed dp/dx vs exact Poiseuille,
    u_max/mean, entry_length_99), obstacles {name: drag split into pressure and viscous, drag_coefficient, lift},
    recirculation {exists, zones: extent, wall {from, to} = separation/reattachment by wall-shear sign change, or
    length_from_obstacle_rear for wakes}, domain (blockage of each obstacle; results depend on it at low Re), scalar,
    transient, evidence {newton_converged, residual, mass_imbalance_rel, mesh_refinement (change of every number between
    two meshes), not_mesh_converged, poiseuille_self_check, verified, reasons}, warnings. Quote a number as verified only
    if evidence.verified; transient numbers are never mesh/time-step verified.

    Example (cylinder D = 1 in a stream, Re = 20): geometry={"mode":"planar","rectangles":[[-30,-30,60,30]],"obstacles":
    [{"name":"cyl","type":"circle","center":[0,0],"radius":0.5}]}, boundaries=[{"name":"in","type":"stream","on":{"side":
    "left"},"velocity":[1,0]},{"name":"top","type":"stream","on":{"side":"top"},"velocity":[1,0]},{"name":"bot","type":
    "stream","on":{"side":"bottom"},"velocity":[1,0]},{"name":"out","type":"outlet","on":{"side":"right"}}], reynolds=20,
    mesh={"boxes":[{"box":[-1.5,-1.5,6,1.5],"size":0.1}]}. Tube with a uniform inlet: geometry={"mode":"axisymmetric",
    "rectangles":[[0,0,0.2,0.005]]}, boundaries=[{"name":"in","type":"inlet","on":{"side":"left"},"mean_velocity":0.01,
    "profile":"uniform"},{"name":"out","type":"outlet","on":{"side":"right"}}], density_kg_m3=1000, viscosity_Pa_s=1e-3."""
    return json.dumps(TF.laminar_flow_2d(geometry, boundaries, density_kg_m3, viscosity_Pa_s, reynolds, reference_length_m,
                                         reference_velocity_m_s, stream_direction, points, sections, pressure_drop_between, scalar, transient,
                                         mesh, figure_path, timeout_s),
                      default=float, ensure_ascii=False)


AGENT_WRAPPERS = [laminar_flow_2d]
