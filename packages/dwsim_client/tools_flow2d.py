"""Agent tool for laminar 2-D incompressible flow on a user-described geometry (planar or axisymmetric), solved with
FEniCSx in a separate process. Engine and evidence: `packages.transport.flow2d`. No DWSIM flowsheet needed; this module
does not import `packages.dwsim_client.tools`."""
from __future__ import annotations

from packages.transport import flow2d as F2


def laminar_flow_2d(geometry: dict, boundaries: list[dict], density_kg_m3: float | None = None,
                    viscosity_Pa_s: float | None = None, reynolds: float | None = None,
                    reference_length_m: float | None = None, reference_velocity_m_s: float | None = None,
                    stream_direction: str = "+x", points: list[list[float]] | None = None,
                    sections: list[dict] | None = None, pressure_drop_between: list[list[str]] | None = None,
                    scalar: dict | None = None, transient: dict | None = None, mesh: dict | None = None,
                    figure_path: str | None = None, timeout_s: int | None = None, rheology: dict | None = None,
                    max_wall_s: float | None = None) -> dict:
    """Laminar incompressible flow on a 2-D geometry YOU describe (steady Navier-Stokes, FEniCSx Taylor-Hood P2-P1 in a
    separate process): channel/tube entry flow, T/L junctions, sudden expansion/contraction, orifice plate, cylinder or
    sphere in a stream. Newtonian or non-Newtonian (power law, Carreau, Carreau-Yasuda via rheology). Optional
    vortex-shedding / start-up transient and passive scalar (Nusselt/Sherwood). Lengths in m.

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
    rheology (non-Newtonian, steady only; give density_kg_m3, NOT viscosity_Pa_s or reynolds; SI units):
      {"model": "power_law", "K" (Pa.s^n), "n", "gamma_reg_1_s"} -> eta = K*(g^2 + gamma_reg^2)^((n-1)/2);
      {"model": "carreau", "eta0", "eta_inf" (default 0), "lambda_s", "n"} -> eta = eta_inf + (eta0-eta_inf)*
      [1+(lambda*g)^2]^((n-1)/2); {"model": "carreau_yasuda", ..., "a" (default 2)} -> [1+(lambda*g)^a]^((n-1)/a).
      g = shear rate sqrt(2 D:D) (axisymmetric: with the u_r/r term). The power law is regularised near g = 0:
      gamma_reg_1_s defaults to 1e-3*U_ref/L_ref and is reported; lower it to check the result does not change. Solved by
      Newton on the full system (start: Newtonian solution with eta(U_ref/L_ref); continuation if needed), tolerances
      and iterations in evidence. Re is then rho*U_ref*L_ref/eta(U_ref/L_ref) (a scale). With a power law, an inlet
      "profile": "parabolic" is the developed power-law profile; use "uniform" for an entry-length problem. No yield
      stress (Bingham) and no viscoelasticity.
    points: [[x,y],...] velocity and pressure there. sections: [{"name","x": c,"y": [lo,hi]}] internal lines (flow, mean p).
      pressure_drop_between: [[a,b],...] boundary names (default: every inlet -> outlet pair).
    scalar: {"peclet" (U_ref*L_ref/D) | "diffusivity_m2_s", "values": {boundary or obstacle name: value}} -> steady
      convection-diffusion with the computed velocity; mean Nu/Sh on each non-inlet surface (zero flux elsewhere).
    transient: {"dt_s","t_end_s","initial","perturbation","min_cycles","stop_when_settled","mesh","refinement",
      "tolerance"} ({} = all defaults) -> semi-implicit BDF2 (coupled u-p, LU reused as preconditioner). "initial":
      "steady" (default: steady solution + perturbation, for vortex shedding) | "rest" (impulsive start from zero
      velocity; use with "stop_when_settled": false and your t_end_s). dt_s default: automatic, 0.05*d/U_ref (d = obstacle
      size, else L_ref) capped by a cell Courant number of 2; reported in transient.time_step. t_end_s default 150
      L_ref/U_ref; the run stops earlier once periodic ("min_cycles", default 8) or steady. "mesh": "base"|"fine".
      "refinement": "dt" (default: repeats with 2*dt on the same mesh) | "mesh" (coarser mesh) | "both" | "none"; the
      changes of strouhal, mean C_D, lift amplitude (or of the point velocities) are the convergence evidence. For a
      periodic flow the time-step check adds a third level (4*dt): each table row gives observed_order,
      extrapolated_dt_to_zero and error_estimate_rel; a row that changes more than the tolerance is accepted only if the
      observed order is 1.5-2.5 (BDF2) and that error estimate is below the tolerance (converged_by says which rule)
      ("tolerance", default 0.02). Without mesh.size / mesh.boxes a wake box of size d/7 is added behind each obstacle.
      A cylinder at Re = 100 takes about 4-9 min (6 000-14 000 cells); set max_wall_s.
    mesh: {"size", "obstacle_size", "growth", "boxes": [{"box": [x0,y0,x1,y1], "size"}], "refine_ratio" (1.5),
      "refinement_check" (true), "tolerance" (0.02)}; give "boxes" in wakes and near re-entrant corners.
    max_wall_s: wall-clock budget (s) of the solver. A transient that does not fit stops cleanly and returns a PARTIAL
      result: "partial": true, "status" explains it, transient.complete = false and its numbers are under
      transient.provisional_not_final -> never quote those as final. timeout_s: hard limit, the process is killed
      (default 300, 1800 with transient, or max_wall_s + margin); without max_wall_s the budget is timeout_s - 20 s.

    Returns Re, reference, boundaries {name: flow_out (per unit depth m2/s, or m3/s axisymmetric), mean_pressure},
    mass_balance, pressure_drops, points, sections, straight_sections (fully developed dp/dx vs the exact Poiseuille or
    power-law value, u_max/mean, entry_length_99 = distance from the section start to where the centre-line velocity
    stays within 1 % of the developed one MEASURED at the section end (entry_length_reference; analytic only if the
    section is not developed), with entry_length_98 / _995 and entry_length_99_sensitivity_rel_per_0p1pct_reference
    (the crossing is flat: quote the length with that sensitivity), entry_length_99_mesh_check {change_rel, converged}
    and entry_length_99_status when NOT mesh converged (then do not quote it), entry_length_99_over_width (tube: /D)),
    rheology {model, parameters, viscosity_law,
    regularisation_shear_rate_1_s, reference_viscosity_Pa_s, Re_metzner_reed (power law in a tube), field: apparent
    viscosity min/max and shear-rate max}, obstacles {name: drag split into pressure and viscous, drag_coefficient,
    lift}, recirculation {exists, zones: extent, wall {from, to} = separation/reattachment by wall-shear sign change, or
    length_from_obstacle_rear for wakes}, domain (blockage d/H and upstream / downstream / lateral distances of each obstacle in obstacle sizes; C_D and St
    depend on the domain at the ~1 % level even at blockage 1-3 %: a warning says so, no blockage correction is applied
    and evidence.verified does NOT cover domain size), scalar,
    transient {state "steady"|"periodic"|"not_settled"|"diverged", complete, stopped_by, strouhal (f*d/U_ref),
    drag_coefficient_mean, lift_coefficient_amplitude (parabolic peak, not the largest sample;
    lift_amplitude_growth_per_cycle = its drift at the end of the run), points_at_end, time_step, refinement {time_step, mesh: table of
    changes, converged}, performance (s/step, LU count)}, partial, status, evidence {newton_converged, residual,
    newton_tolerances, mass_imbalance_rel, mesh_refinement (change of every steady number between two meshes),
    not_mesh_converged, poiseuille_self_check, transient_time_step_converged, transient_mesh_converged, verified,
    reasons}, warnings. Quote a number as verified only if evidence.verified; a transient is verified only when complete
    and BOTH its time-step and mesh checks pass ("refinement": "both"); otherwise say which check is missing.

    Example (cylinder D = 1 in a stream, Re = 20): geometry={"mode":"planar","rectangles":[[-30,-30,60,30]],"obstacles":
    [{"name":"cyl","type":"circle","center":[0,0],"radius":0.5}]}, boundaries=[{"name":"in","type":"stream","on":{"side":
    "left"},"velocity":[1,0]},{"name":"top","type":"stream","on":{"side":"top"},"velocity":[1,0]},{"name":"bot","type":
    "stream","on":{"side":"bottom"},"velocity":[1,0]},{"name":"out","type":"outlet","on":{"side":"right"}}], reynolds=20,
    mesh={"boxes":[{"box":[-1.5,-1.5,6,1.5],"size":0.1}]}. Tube with a uniform inlet: geometry={"mode":"axisymmetric",
    "rectangles":[[0,0,0.2,0.005]]}, boundaries=[{"name":"in","type":"inlet","on":{"side":"left"},"mean_velocity":0.01,
    "profile":"uniform"},{"name":"out","type":"outlet","on":{"side":"right"}}], density_kg_m3=1000, viscosity_Pa_s=1e-3.
    Same tube with a shear-thinning fluid (entry length): drop viscosity_Pa_s and add rheology={"model":"power_law","K":
    0.01,"n":0.5}. Vortex shedding: the cylinder above with reynolds=100, the right side {"type":"outlet","normal_flow":
    false}, transient={"refinement":"both"}, max_wall_s=900."""
    return F2.laminar_flow_2d(geometry, boundaries, density_kg_m3, viscosity_Pa_s, reynolds, reference_length_m,
                              reference_velocity_m_s, stream_direction, points, sections, pressure_drop_between, scalar, transient,
                              mesh, figure_path, timeout_s, rheology, max_wall_s)


FLOW2D_TOOLS = [laminar_flow_2d]
