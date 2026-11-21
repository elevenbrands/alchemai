"""generalized_compressibility (Lee-Kesler) against CoolProp's reference equations of state.

The Lee-Kesler constants in packages/properties/lee_kesler.py were written from memory; a wrong digit in any of them
would push Z off by far more than the few percent a generalized correlation is known to deviate. CoolProp (Helmholtz
reference EOS for each fluid) is the independent check; the critical constants and ω are CoolProp's own so that only the
correlation is tested.
"""
import pytest
import CoolProp.CoolProp as CP

from packages.dwsim_client import tools_properties as TPR

FLUIDS = {"Argon": "argon", "Methane": "methane", "Nitrogen": "nitrogen", "Propane": "propane", "CarbonDioxide": "carbon dioxide"}


def _crit(fl):
    return {"Tc_K": CP.PropsSI("Tcrit", fl), "Pc_Pa": CP.PropsSI("Pcrit", fl), "omega": CP.PropsSI("acentric", fl)}


@pytest.mark.parametrize("fl", list(FLUIDS))
@pytest.mark.parametrize("Tr,Pr,tol", [(1.2, 1.0, 0.015), (1.5, 2.0, 0.015), (1.1, 0.5, 0.01), (2.0, 5.0, 0.03), (3.0, 8.0, 0.04)])
def test_matches_reference_eos(fl, Tr, Pr, tol):
    c = _crit(fl)
    z = CP.PropsSI("Z", "T", Tr * c["Tc_K"], "P", Pr * c["Pc_Pa"], fl)
    o = TPR.generalized_compressibility([FLUIDS[fl]], T_K=Tr * c["Tc_K"], P_Pa=Pr * c["Pc_Pa"], critical={FLUIDS[fl]: c})
    assert o["ok"] and o["Z"] == pytest.approx(z, rel=tol)


def test_simple_fluid_limits_and_inverse_modes():
    o = TPR.generalized_compressibility(["methane"], T_K=300.0, P_Pa=1e3)
    assert o["Z"] == pytest.approx(1.0, abs=1e-3)                       # dilute gas
    o = TPR.generalized_compressibility(["methane"], T_K=250.0, P_Pa=8e6, moles=10.0)
    v = o["molar_volume_m3_mol"]
    assert o["volume_m3"] == pytest.approx(10 * v)
    back_p = TPR.generalized_compressibility(["methane"], T_K=250.0, molar_volume_m3_mol=v)
    back_t = TPR.generalized_compressibility(["methane"], P_Pa=8e6, volume_m3=10 * v, moles=10.0)
    assert back_p["P_Pa"] == pytest.approx(8e6, rel=1e-7) and back_t["T_K"] == pytest.approx(250.0, rel=1e-7)


def test_kay_rule_and_newton_correction():
    mix = TPR.generalized_compressibility(["methane", "ethane"], T_K=300.0, P_Pa=5e6, mole_fractions=[0.6, 0.4])
    pc = mix["pseudo_critical"]
    rows = pc["components"]
    assert pc["Tc_K"] == pytest.approx(0.6 * rows[0]["Tc_K"] + 0.4 * rows[1]["Tc_K"]) and "Kay" in pc["rule"]
    zc = CP.PropsSI("Z", "T", 100.0, "P", 5e6, "Hydrogen")
    chart = TPR.generalized_compressibility(["hydrogen"], T_K=100.0, P_Pa=5e6, chart="two_parameter")
    raw = TPR.generalized_compressibility(["hydrogen"], T_K=100.0, P_Pa=5e6, chart="two_parameter", quantum_gas_correction=False)
    assert chart["pseudo_critical"]["Tc_K"] == pytest.approx(raw["pseudo_critical"]["Tc_K"] + 8.0)
    assert chart["pseudo_critical"]["omega"] == 0.0
    assert abs(chart["Z"] - zc) < abs(raw["Z"] - zc) and chart["Z"] == pytest.approx(zc, rel=0.02)   # Newton helps the 2-parameter chart
    lk = TPR.generalized_compressibility(["hydrogen"], T_K=100.0, P_Pa=5e6)                            # 3-parameter: no correction
    assert lk["pseudo_critical"]["Tc_K"] == pytest.approx(raw["pseudo_critical"]["Tc_K"]) and lk["Z"] == pytest.approx(zc, rel=0.01)
    assert "error" in TPR.generalized_compressibility(["methane"], T_K=300.0)
    assert "error" in TPR.generalized_compressibility(["methane"], T_K=300.0, P_Pa=1e5, molar_volume_m3_mol=0.02)
