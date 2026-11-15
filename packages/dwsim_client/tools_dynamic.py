"""Agent tool for dynamic simulation with DWSIM's own dynamic mode (arranque, paro, transitorios).
The engine and its evidence live in `packages.dwsim_client.dynamic`; DWSIM is loaded only when the tool is called."""
from __future__ import annotations

from packages.dwsim_client import dynamic as DY


def dwsim_dynamic_run(compounds: list[str], property_package: str, streams: list[dict], units: list[dict], step_s: float,
                      duration_s: float, monitors: list[dict], events: list[dict] | None = None,
                      settle_s: float | None = None, check_step_halving: bool = True, step_tolerance: float = 0.01,
                      max_points: int = 400) -> dict:
    """Simulación dinámica con el modo dinámico de DWSIM (integrador propio de DWSIM, Euler explícito de paso fijo) de un
    flowsheet de tanques abiertos, válvulas de líquido con Kv y controladores PID: arranque (llenado), paro (corte de
    alimentación), y respuesta a escalones de caudal, temperatura, composición, apertura de válvula o consigna. Todo en SI.

    compounds: nombres de DWSIM (p. ej. "Water", "Ethanol"). property_package: p. ej. "SteamTables", "NRTL", "PengRobinson".
    streams: lista de {name, dynamic_spec, ...}. DWSIM exige en cada corriente dynamic_spec "flow" o "pressure":
      · alimentación con caudal dado: dynamic_spec "flow", temperature_K, pressure_Pa, mass_flow_kg_s (o molar_flow_mol_s),
        mole_fractions (si hay más de un compuesto);
      · alimentación desde un cabezal a presión que entra por una válvula: dynamic_spec "pressure", temperature_K, pressure_Pa,
        mole_fractions y un mass_flow_kg_s de arranque compatible con el Kv (después el caudal lo calcula la válvula);
      · salida de un tanque: solo {name, dynamic_spec: "pressure"} (la presión la fija el nivel) y debe ir a una válvula;
      · salida de una válvula: dynamic_spec "pressure" y pressure_Pa = presión aguas abajo (101325 si descarga a un tanque
        abierto o a la atmósfera).
    units: lista de equipos, cada uno con kind y name:
      · tank: feed, product (nombres de corriente), volume_m3, height_m (área = volumen/altura), initial_level_m, e
        initial_content = "inlet" (mismo fluido que su entrada) o {temperature_K, mole_fractions} si initial_level_m > 0;
      · valve: feed, product, kv (m³/h de agua con 1 bar, válvula abierta), opening_pct (0-100 en t = 0), characteristic
        ("linear", "equal_percentage" con rangeability, o "quick_opening");
      · pid: controlled {object, variable}, manipulated {object, variable}, setpoint, kp, ki (1/s), kd (s), reverse_acting
        (true si la manipulada debe subir cuando la medida supera la consigna, p. ej. nivel con válvula de salida),
        output_bias, output_span, output_min, output_max (PID de DWSIM: error = (medida − consigna)/|consigna inicial|;
        manipulada = output_bias ± (Kp·e + Ki·∫e dt + Kd·de/dt)·output_span).
    monitors: lista de {object, variable}. Variables de corriente: temperature_K, pressure_Pa, mass_flow_kg_s,
      molar_flow_mol_s, density_kg_m3, specific_enthalpy_kJ_kg, mole_fraction:<compuesto>, mass_fraction:<compuesto>;
      de tanque: level_m; de válvula: opening_pct, pressure_drop_Pa; de pid: setpoint, output.
    events: cambios escalón {time_s, object, variable, value} sobre alimentaciones (temperature_K, pressure_Pa,
      mass_flow_kg_s, molar_flow_mol_s), válvulas (opening_pct) o pid (setpoint); o {time_s, object, mole_fractions} para
      cambiar la composición de una alimentación. Paro = caudal de alimentación a 0; arranque = initial_level_m 0.
    step_s, duration_s: paso y horizonte. settle_s: integra ese tiempo sin eventos antes de t = 0 y comprueba que el estado
      ya no cambia (inicialización en estado estacionario). check_step_halving repite el caso con step_s/2.

    Nada se inventa: volúmenes, alturas, niveles iniciales, Kv, presiones de contorno y ajustes del PID son datos obligatorios.
    Devuelve time_s, series {"objeto.variable": [...]}, units, initial, final, minimum, maximum, events_applied, tanks (inventario
    inicial y final y cierre del balance), evidence (verified = pasada estacionaria resuelta, integración completa, balance de
    materia de cada tanque cerrado con la regla explícita de DWSIM, independencia del paso y, si se pidió, estado inicial
    asentado; reasons si no), warnings (tanque vacío o nivel sobre la altura), assumptions, not_included y method."""
    return DY.run_dynamic(compounds, property_package, streams, units, step_s, duration_s, monitors, events, settle_s,
                          check_step_halving, step_tolerance, max_points=max_points)


DYNAMIC_TOOLS = [dwsim_dynamic_run]
