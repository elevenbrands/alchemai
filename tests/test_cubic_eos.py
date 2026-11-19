"""cubic_eos_volume: single-phase molar volume from van der Waals, Redlich-Kwong, SRK and Peng-Robinson.

Independent references: the RK, SRK, PR and VDW classes of the `thermo` library (a separate implementation), the
critical-point identities of each equation (Zc = 3/8, 1/3, 1/3, 0.3074) and the ideal-gas limit."""
import math

import pytest
from thermo import eos

from packages.dwsim_client import tools_properties as T
from packages.properties import cubic_eos as CE

ATM = 101325.0
THERMO = {"RK": eos.RK, "SRK": eos.SRK, "PR": eos.PR, "VDW": eos.VDW}
CASES = [  # Tc K, Pc Pa, omega, T K, P Pa
    (405.5, 111.3 * ATM, 0.25, 450.0, 56 * ATM),
    (512.6, 79.9 * ATM, 0.559, 573.15, 100 * ATM),
    (369.83, 4.248e6, 0.152, 300.0, 5e5),
    (190.56, 4.599e6, 0.011, 250.0, 8e6),
    (647.1, 22.064e6, 0.344, 500.0, 1e5),
]


def _user(name, tc, pc, om):
    return [{"compound": name, "Tc_K": tc, "Pc_Pa": pc, "omega": om}]


@pytest.mark.parametrize("model", ["RK", "SRK", "PR", "VDW"])
@pytest.mark.parametrize("tc,pc,om,t,p", CASES)
def test_pure_roots_match_thermo(model, tc, pc, om, t, p):
    r = T.cubic_eos_volume(["nitrogen"], model, t, p, critical_constants=_user("nitrogen", tc, pc, om))
    kw = {"Tc": tc, "Pc": pc, "T": t, "P": p}
    if model in CE.NEEDS_OMEGA:
        kw["omega"] = om
    e = THERMO[model](**kw)
    ref = sorted(v for v in (getattr(e, "V_l", None), getattr(e, "V_g", None)) if v is not None)
    got = [x["V_m3_mol"] for x in r["roots"] if not x["type"].startswith("unstable")]
    assert len(got) == len(ref)
    for g, v in zip(got, ref):
        assert g == pytest.approx(v, rel=1e-8)
    assert r["evidence"]["verified"] and r["evidence"]["max_pressure_closure_rel"] < 1e-9


@pytest.mark.parametrize("model", ["SRK", "PR"])
def test_stable_root_follows_the_saturation_pressure_of_the_equation(model):
    tc, pc, om, t = 369.83, 4.248e6, 0.152, 300.0
    psat = THERMO[model](Tc=tc, Pc=pc, omega=om, T=t, P=1e5).Psat(t)
    below = T.cubic_eos_volume(["propane"], model, t, psat * 0.98, critical_constants=_user("propane", tc, pc, om))
    above = T.cubic_eos_volume(["propane"], model, t, psat * 1.02, critical_constants=_user("propane", tc, pc, om))
    assert below["root_type"] == "gas-like" and above["root_type"] == "liquid-like"
    assert below["root_is_lowest_gibbs"] and above["root_is_lowest_gibbs"]
    forced = T.cubic_eos_volume(["propane"], model, t, psat * 1.02, critical_constants=_user("propane", tc, pc, om),
                                root="gas")
    assert forced["root_type"] == "gas-like" and not forced["root_is_lowest_gibbs"]
    assert any("metaestable" in w for w in forced["warnings"])


@pytest.mark.parametrize("model,zc", [("VDW", 0.375), ("RK", 1 / 3), ("SRK", 1 / 3), ("PR", 0.30740)])
def test_critical_compressibility(model, zc):
    r = T.cubic_eos_volume(["methane"], model, 190.56, 4.599e6, critical_constants=_user("methane", 190.56, 4.599e6, 0.011))
    assert r["Z"] == pytest.approx(zc, abs=2e-3)       # triple root at the critical point: ill-conditioned by nature


def test_ideal_gas_limit_and_units():
    r = T.cubic_eos_volume(["nitrogen"], "RK", 400.0, 10.0)
    assert r["Z"] == pytest.approx(1.0, abs=1e-6)
    assert r["V_L_mol"] == pytest.approx(r["V_m3_mol"] * 1e3) and r["V_m3_kmol"] == pytest.approx(r["V_m3_mol"] * 1e3)
    assert r["ideal_gas"]["V_m3_mol"] == pytest.approx(CE.R_GAS * 400.0 / 10.0)
    assert r["components"][0]["Tc_K"]["source"] == "library" and r["status"] == "ok"
    for k in ("warnings", "assumptions", "not_included", "method", "evidence"):
        assert k in r


@pytest.mark.parametrize("model", ["RK", "SRK", "PR"])
def test_mixture_matches_thermo_mix(model):
    names = ["carbon monoxide", "water", "carbon dioxide", "hydrogen"]
    zs = [630 / 2012, 1130 / 2012, 189 / 2012, 63 / 2012]
    r = T.cubic_eos_volume(names, model, 500.0, 200 * ATM, zs, root="gas")
    cs = r["components"]
    cls = {"RK": eos_mix_rk, "SRK": eos_mix_srk, "PR": eos_mix_pr}[model]
    v = cls([c["Tc_K"]["value"] for c in cs], [c["Pc_Pa"]["value"] for c in cs],
            [c.get("omega", {}).get("value", 0.0) for c in cs], zs, 500.0, 200 * ATM)
    assert r["V_m3_mol"] == pytest.approx(v, rel=1e-8)
    assert all(b["source"] == "zero_by_absence" for b in r["binary_parameters"])
    assert any("equilibrio de fases" in w for w in r["warnings"])


def eos_mix_rk(tcs, pcs, oms, zs, t, p):
    from thermo.eos_mix import RKMIX
    e = RKMIX(Tcs=tcs, Pcs=pcs, zs=zs, T=t, P=p)
    return max(v for v in (getattr(e, "V_l", None), getattr(e, "V_g", None)) if v is not None)


def eos_mix_srk(tcs, pcs, oms, zs, t, p):
    from thermo.eos_mix import SRKMIX
    e = SRKMIX(Tcs=tcs, Pcs=pcs, omegas=oms, zs=zs, T=t, P=p)
    return max(v for v in (getattr(e, "V_l", None), getattr(e, "V_g", None)) if v is not None)


def eos_mix_pr(tcs, pcs, oms, zs, t, p):
    from thermo.eos_mix import PRMIX
    e = PRMIX(Tcs=tcs, Pcs=pcs, omegas=oms, zs=zs, T=t, P=p)
    return max(v for v in (getattr(e, "V_l", None), getattr(e, "V_g", None)) if v is not None)


def test_kij_changes_the_mixture_and_is_reported():
    names, zs = ["methane", "carbon dioxide"], [0.5, 0.5]
    r0 = T.cubic_eos_volume(names, "PR", 250.0, 5e6, zs)
    r1 = T.cubic_eos_volume(names, "PR", 250.0, 5e6, zs, kij=[{"i": "methane", "j": "CO2", "kij": 0.1}])
    assert r1["binary_parameters"][0] == {"i": r1["components"][0]["compound"], "j": r1["components"][1]["compound"],
                                          "kij": 0.1, "source": "user"}
    assert r1["V_m3_mol"] > r0["V_m3_mol"]            # weaker attraction -> larger volume
    from thermo.eos_mix import PRMIX
    cs = r1["components"]
    e = PRMIX(Tcs=[c["Tc_K"]["value"] for c in cs], Pcs=[c["Pc_Pa"]["value"] for c in cs],
              omegas=[c["omega"]["value"] for c in cs], zs=zs, T=250.0, P=5e6, kijs=[[0, 0.1], [0.1, 0]])
    assert r1["V_m3_mol"] == pytest.approx(e.V_g, rel=1e-8)


def test_input_errors_are_spanish_and_specific():
    with pytest.raises(ValueError, match="No hay valor por defecto"):
        T.cubic_eos_volume(["ammonia"], "", 400.0, 1e5)
    with pytest.raises(ValueError, match="no disponible"):
        T.cubic_eos_volume(["ammonia"], "BWR", 400.0, 1e5)
    with pytest.raises(ValueError, match="mole_fractions"):
        T.cubic_eos_volume(["ammonia", "water"], "RK", 400.0, 1e5)
    with pytest.raises(ValueError, match="suma"):
        T.cubic_eos_volume(["ammonia", "water"], "RK", 400.0, 1e5, [0.5, 0.6])
    with pytest.raises(ValueError, match="no está en components"):
        T.cubic_eos_volume(["ammonia"], "RK", 400.0, 1e5, critical_constants=[{"compound": "water", "Tc_K": 647.0}])
    with pytest.raises(ValueError, match="root"):
        T.cubic_eos_volume(["ammonia"], "RK", 400.0, 1e5, root="vapor")
    r = T.cubic_eos_volume(["ammonia"], "RKS", 400.0, 1e5)
    assert r["model"] == "SRK" and math.isfinite(r["Z"])
