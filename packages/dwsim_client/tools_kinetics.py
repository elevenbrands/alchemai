"""Agent tools for detailed kinetics and combustion (Cantera, in process, BSD-3): adiabatic flame and reactor networks.
They need no DWSIM flowsheet. The engine and its evidence live in `packages.kinetics.cantera_engine`."""
from __future__ import annotations

from packages.kinetics import cantera_engine as CE


def adiabatic_flame(fuel: str, equivalence_ratio: float, T_K: float = 298.15, P_Pa: float = 101325.0,
                    oxidizer: str = "O2:1, N2:3.76", mechanism: str = "gri30.yaml", laminar_flame_speed: bool = False,
                    report_species: list[str] | None = None) -> dict:
    """Adiabatic flame of a premixed fuel/oxidizer mixture with Cantera (default GRI-Mech 3.0, 53 species, natural gas).
    fuel and oxidizer are compositions in mechanism species names ('CH4', 'CH4:0.9, C2H6:0.1', 'H2'; air = 'O2:1, N2:3.76').
    Returns T_adiabatic_equilibrium_K (with dissociation: CO, H2, OH, NO...), T_adiabatic_complete_combustion_K (no
    dissociation, what a 100 % conversion reactor gives: an UPPER bound; None if rich), dissociation_penalty_K, the
    equilibrium products, burnt/unburnt densities and, with laminar_flame_speed=True (~20 s), the laminar flame speed
    (1-D free flame, checked on a finer grid). evidence.verified only if elements and enthalpy are conserved, the
    thermo range holds and (for flame speed) the grid change < 2 % and the GRI-Mech validation range (1000-2500 K,
    10 Torr-10 atm, φ 0.1-5) holds."""
    return CE.adiabatic_flame(fuel, equivalence_ratio, T_K, P_Pa, oxidizer, mechanism, laminar_flame_speed, report_species)


def kinetic_reactor_network(reactor: str, feed: str, T_K: float, P_Pa: float, mechanism: str = "gri30.yaml",
                            energy: str = "adiabatic", end_time_s: float | None = None,
                            residence_times_s: list[float] | None = None, key_species: list[str] | None = None,
                            report_species: list[str] | None = None, initial_state: str = "ignited",
                            custom_species: list[dict] | None = None, custom_reactions: list[dict] | None = None) -> dict:
    """Reactor on a detailed mechanism (Cantera, default GRI-Mech 3.0) or on your power-law kinetics.
    reactor: 'batch_constant_volume' | 'batch_constant_pressure' | 'pfr' (ideal, lagrangian at constant P; end_time_s =
    residence time) | 'wsr' (well-stirred reactor / CSTR at steady state for each of residence_times_s, 1-20 values).
    feed: mole composition in mechanism species ('CH4:1, O2:2, N2:7.52'); T_K, P_Pa: feed state (batch initial state).
    energy: 'adiabatic' | 'isothermal'. Batch/PFR return the final state, a profile and, if adiabatic, the ignition delay
    (max dT/dt). WSR returns T, conversions, CO/NO per τ, ignited/extinguished state and blow_out_between_s;
    initial_state 'ignited' starts from equilibrium (upper branch). key_species: species for conversion (default: feed
    species except O2 and inerts). Custom kinetics (isothermal only): custom_species [{'name': 'A', 'composition':
    {'C': 4, 'H': 10}}, ...] and custom_reactions [{'equation': 'A => B', 'A': 1e8, 'b': 0, 'Ea_J_mol': 1e5,
    'orders': {'A': 1}}] with concentrations in mol/m³ (irreversible only). evidence.verified only if elements (and
    energy if adiabatic) are conserved, a 1000× looser integrator tolerance changes nothing, the WSR is at steady state
    and the mechanism's validation range holds."""
    return CE.kinetic_reactor_network(reactor, feed, T_K, P_Pa, mechanism, energy, end_time_s, residence_times_s, key_species,
                                      report_species, initial_state, custom_species, custom_reactions)


KINETICS_TOOLS = [adiabatic_flame, kinetic_reactor_network]
