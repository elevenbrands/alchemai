"""Notebook 33 — El diagnóstico de vapor contra mediciones PUBLICADAS de calderas reales (bloque M22).

No son clientes: son calderas medidas por terceros y publicadas. Sirven para lo que los ejemplos del DOE no dan: datos de
campo, con sus huecos y sus inconsistencias. Fuentes leídas el 2026-11-14 (los números de cada caso se cotejaron con el
texto o la tabla de la fuente):

A. México, gas natural, caldera de tubos de humo de 600 C.C. a 2 250 m s. n. m. E. L. Martínez Obregón, «Diagnóstico
   energético del sistema de vapor en una fábrica de alimentos», tesina, Facultad de Ingeniería (PUEI), UNAM, 2021.
   https://ru.dgb.unam.mx/bitstream/20.500.14330/TES01000813111/3/0813111.pdf  Analizador Bacharach PCA3, 5 lecturas.
B. Argentina (Tucumán), gas natural, caldera acuotubular de 48 t/h. M. A. Golato y otros, «Metodología de cálculo de la
   eficiencia térmica de generadores de vapor», Rev. Ind. y Agríc. de Tucumán 85(2): 17-31, 2008, Tabla 6 B (promedio de
   cinco ensayos). http://www.scielo.org.ar/pdf/riat/v85n2/v85n2a03.pdf
C. Ejemplo resuelto de combustóleo de la Bureau of Energy Efficiency (India), «Energy Performance Assessment of
   Boilers», sección 1.8. NO es una caldera identificada: se usa porque da el desglose pérdida por pérdida.

Lo que la fuente no da y el motor exige se declara en `NOT_IN_SOURCE` y nunca entra en una comparación.

    PYTHONPATH=. uv run python -u notebooks/33_calderas_reales_publicadas.py
"""
from __future__ import annotations

import json
from pathlib import Path

from CoolProp.CoolProp import PropsSI

from packages.energy import combustion as CB
from packages.energy import fuel_reference as FR
from packages.energy import steam_diagnostic as SD

HERE = Path(__file__).resolve().parent
C = 273.15
NOT_IN_SOURCE = {
    "A": ["composición del gas (se usa el gas de referencia GULF-COAST y se informa la sensibilidad)",
          "humedad del aire (se corre con 0 y con 60 % de humedad relativa)",
          "temperatura del gas en el quemador (se toma la del aire)",
          "precio del agua (hace falta para correr el diagnóstico completo; no entra en ninguna comparación)",
          "base del poder calorífico facturado (la tesina lo llama una vez «PCI»; se discute abajo)"],
    "B": ["humedad del aire (0 y 60 %)", "temperatura del gas en el quemador (la del aire)",
          "pérdida por radiación (la fuente la toma de una carta ASME y no imprime el valor)"],
    "C": [],
}


def case_a() -> dict:
    """UNAM 2021. Measured: O2 2.7 %, CO2 10.28 %, CO 0 ppm, stack 211, air 22.52 (°C), analyzer %EFF 82.26, %EA 13.18.
    Direct method of the source: 3 345.81 kg/h x (2763.9 - 230.2) kJ/kg over 286.81 m3/h x 36 636 kJ/m3 = 80.68 %."""
    P_site = SD.standard_atmosphere_Pa(2250.0)
    air_T, stack_T = 22.52 + C, 211.0 + C
    out: dict = {"source_values": {"analyzer_efficiency": 0.8226, "analyzer_excess_air": 0.1318, "CO2_dry": 0.1028,
                                    "direct_efficiency": 0.8068}, "site_pressure_Pa": P_site}
    runs = {}
    for gas in ("GULF-COAST", "AMARILLO", "NATURAL-GAS-SAMPLE"):
        for rh in (0.0, 0.6):
            r = CB.combustion_efficiency(
                {"type": "gas", "composition": FR.reference_gas(gas)["composition"], "T_K": air_T},
                {"T_K": air_T, "relative_humidity": rh}, P_site, flue_O2={"mole_fraction": 0.027, "basis": "dry"},
                stack_T_K=stack_T, setting_loss_fraction=0.0, setting_loss_basis="HHV", adiabatic_flame=False)
            runs[f"{gas}, HR {int(100 * rh)} %"] = {
                "efficiency_HHV": r["efficiency"]["HHV"]["efficiency"], "efficiency_LHV": r["efficiency"]["LHV"]["efficiency"],
                "excess_air": r["air"]["excess_air_fraction"], "CO2_dry": r["flue_gas"]["dry_mole_fractions"]["CO2"]}
    out["combustion_runs"] = runs
    ref = runs["GULF-COAST, HR 60 %"]
    hhv = [v["efficiency_HHV"] for v in runs.values()]
    out["comparison"] = {
        "combustion_efficiency_HHV": {"engine": ref["efficiency_HHV"], "engine_range": [min(hhv), max(hhv)], "source": 0.8226,
                                      "difference_points": 100 * (ref["efficiency_HHV"] - 0.8226)},
        "combustion_efficiency_LHV_for_reference": ref["efficiency_LHV"],
        "excess_air": {"engine": ref["excess_air"], "source": 0.1318, "difference_points": 100 * (ref["excess_air"] - 0.1318)},
        "CO2_dry": {"engine": ref["CO2_dry"], "source": 0.1028, "difference_points": 100 * (ref["CO2_dry"] - 0.1028)},
    }
    # the source's direct method, redone with IAPWS-95 instead of its printed enthalpies
    P_steam = 7.05e5                                                              # «vapor saturado a 7.05 bara»
    h_steam, h_fw = PropsSI("H", "P", P_steam, "Q", 1, "Water"), PropsSI("H", "T", 55.0 + C, "Q", 0, "Water")
    fuel_kJ_h = 286.81 * 36636.19
    out["direct_method"] = {"source_enthalpies_kJ_kg": [2763.9, 230.2], "iapws_enthalpies_kJ_kg": [h_steam / 1e3, h_fw / 1e3],
                            "with_source_enthalpies": 3345.81 * (2763.9 - 230.2) / fuel_kJ_h,
                            "with_iapws": 3345.81 * (h_steam - h_fw) / 1e3 / fuel_kJ_h, "source": 0.8068}
    # the full diagnostic, as a plant would get it: gas without analysis, setting loss unknown, capacity 600 C.C.
    hv_basis = "HHV"
    d = SD.steam_system_diagnostic(
        "MXN", fuel={"type": "natural_gas_unknown", "T_K": air_T},
        combustion={"site_altitude_m": 2250.0, "air": {"T_K": air_T, "relative_humidity": 0.6},
                    "flue_O2": {"mole_fraction": 0.027, "basis": "dry"}, "flue_CO_dry_mole_fraction": 0.0, "stack_T_K": stack_T,
                    "setting_loss_unknown": True},
        boiler={"steam_pressure_Pa": P_steam, "blowdown_fraction": 0.02, "rated_steam_capacity_kg_s": 600 * 15.65 / 3600.0},
        water={"feedwater_heating": "not_charged", "feedwater_T_K": 55.0 + C, "makeup_water_T_K": 20.0 + C,
               "condensate_return_fraction": 0.0},
        fuel_cost={"price_per_GJ": 122.4796, "basis": hv_basis}, water_cost={"price_per_m3": 1.0},      # water: NOT_IN_SOURCE
        operation={"hours_per_year": 8760.0, "steam_generated_kg_s": 3345.81 / 3600.0, "annual_basis": "steam",
                   "fuel_consumption": {"energy_GJ_per_year": fuel_kJ_h * 8760.0 / 1e6, "basis": hv_basis}},
        today="2026-11-14")
    x = d["evidence"]["input_output_cross_check"]
    out["diagnostic"] = {
        "boiler_efficiency_by_losses": d["baseline"]["boiler"]["efficiency"], "setting_loss_used": d["setting_loss"]["value"],
        "load_fraction": d["baseline"]["load_fraction"], "input_output_efficiency": x["input_output_efficiency"],
        "difference_points": x["difference_points"], "source_direct_efficiency": 0.8068,
        "data_flags": [f["id"] for f in d["data_review"]["flags"]], "verified": d["evidence"]["verified"],
        "assumed": [v["id"] for v in d["assumed_values"]]}
    out["reading"] = ("El analizador no dice en qué base da su eficiencia. En base PCS el motor queda a décimas de punto; en base "
                      "PCI estaría unos 9 puntos arriba. La eficiencia directa de la tesina (80.68 %) solo es coherente con "
                      "la del analizador (82.26 %) si el poder calorífico facturado es el superior, así que se lee como PCS.")
    return out


def case_b() -> dict:
    """Tucumán 2008, Table 6 B: O2 1.8 %, CO2 11.01 %, CO 0.015 % (dry), lambda 1.08, stack 316.9 °C, air 35.9 °C, steam
    22 200 kg/h at 17.5 ata and 316.9 °C, feed water 106.8 °C, gas 2 156 Nm3/h. Source: direct (ASME) 74.4 % on PCI."""
    comp = {"N2": 0.0066, "CO2": 0.0222, "CH4": 0.9039, "C2H6": 0.0535, "C3H8": 0.0089, "n-C4H10": 0.0033,
            "n-C5H12": 0.0011 + 0.0005}                                           # C6H14 0.05 % counted as pentane
    air_T, stack_T = 35.9 + C, 316.9 + C
    runs = {}
    for rh in (0.0, 0.6):
        r = CB.combustion_efficiency({"type": "gas", "composition": comp, "T_K": air_T}, {"T_K": air_T, "relative_humidity": rh},
                                     101325.0, flue_O2={"mole_fraction": 0.018, "basis": "dry"},
                                     flue_CO_dry_mole_fraction=0.00015, stack_T_K=stack_T, setting_loss_fraction=0.0,
                                     setting_loss_basis="LHV", adiabatic_flame=False)
        runs[f"HR {int(100 * rh)} %"] = r
    r = runs["HR 60 %"]
    P = 17.5 * 98066.5
    h_steam, h_fw = PropsSI("H", "P", P, "T", 316.9 + C, "Water"), PropsSI("H", "T", 106.8 + C, "Q", 0, "Water")
    duty_W = 22200.0 / 3600.0 * (h_steam - h_fw)
    fuel_LHV_W = 2156.0 / 3600.0 * r["fuel"]["LHV_J_Nm3"]
    fuel_HHV_W = 2156.0 / 3600.0 * r["fuel"]["HHV_J_Nm3"]
    by_losses = r["efficiency"]["LHV"]["efficiency"]
    return {
        "source_values": {"CO2_dry": 0.1101, "lambda": 1.08, "direct_efficiency_LHV": 0.744, "balance_efficiency_LHV": 0.767},
        "comparison": {
            "CO2_dry": {"engine": r["flue_gas"]["dry_mole_fractions"]["CO2"], "source": 0.1101,
                        "difference_points": 100 * (r["flue_gas"]["dry_mole_fractions"]["CO2"] - 0.1101)},
            "lambda": {"engine": 1.0 + r["air"]["excess_air_fraction"], "source": 1.08,
                       "difference": 1.0 + r["air"]["excess_air_fraction"] - 1.08},
            "direct_efficiency_LHV": {"engine": duty_W / fuel_LHV_W, "source": 0.744,
                                      "difference_points": 100 * (duty_W / fuel_LHV_W - 0.744)},
            "direct_efficiency_HHV": duty_W / fuel_HHV_W,
            "efficiency_by_losses_LHV_without_radiation": {"engine": by_losses,
                                                           "range_with_humidity": [runs["HR 0 %"]["efficiency"]["LHV"]["efficiency"],
                                                                                   by_losses]},
            "losses_minus_direct_points": 100 * (by_losses - duty_W / fuel_LHV_W)},
        "heating_values": {"LHV_MJ_Nm3": r["fuel"]["LHV_J_Nm3"] / 1e6, "HHV_MJ_Nm3": r["fuel"]["HHV_J_Nm3"] / 1e6},
        "reading": ("Con la composición dada y el O2 medido, el CO2 y el exceso de aire del motor coinciden con los medidos. La "
                    "eficiencia directa rehecha con IAPWS-95 coincide con la de la fuente. La de pérdidas queda muy por encima "
                    "de la directa: con una diferencia así el diagnóstico marca «efficiency_gap» en vez de dar un número por "
                    "bueno. La tabla de la fuente da la misma cifra, 316.9, para la temperatura del vapor y la de los gases."),
    }


def case_c() -> dict:
    """BEE worked example (fuel oil): L1 7.86, L2 7.08, L3 0.033, L4 0.38, L6 0.38 %, efficiency 84.27 % on GCV."""
    fuel = {"type": "ultimate", "mass_fractions": {"C": 0.84, "H": 0.12, "N": 0.005, "O": 0.015, "S": 0.015, "moisture": 0.005},
            "HHV_J_kg": 10000 * 4186.8, "T_K": 30.0 + C}
    r = CB.combustion_efficiency(fuel, {"T_K": 30.0 + C, "humidity_ratio_kg_kg": 0.025}, 101325.0,
                                 flue_O2={"mole_fraction": 0.074, "basis": "dry"}, stack_T_K=190.0 + C,
                                 setting_loss_fraction=0.0038, setting_loss_basis="HHV", adiabatic_flame=False)
    L = r["efficiency"]["HHV"]["losses_fraction_of_heating_value"]
    credits = r["efficiency"]["HHV"]["credits_fraction_of_heating_value"]
    # the engine counts the latent heat of the fuel moisture together with that of the water formed, so L2 and L3 go together
    mine = {"L1 gases secos": 100 * L["dry_gas"],
            "L2 + L3 agua del hidrógeno y humedad del combustible":
                100 * (L["water_latent"] + L["water_from_hydrogen_sensible"] + L.get("fuel_moisture_sensible", 0.0)),
            "L4 humedad del aire": 100 * L["air_moisture_sensible"], "L6 radiación": 100 * L["setting"]}
    source = {"L1 gases secos": 7.86, "L2 + L3 agua del hidrógeno y humedad del combustible": 7.08 + 0.033,
              "L4 humedad del aire": 0.38, "L6 radiación": 0.38}
    return {"losses_percent": {k: {"engine": mine[k], "source": source[k], "difference_points": mine[k] - source[k]} for k in mine},
            "credits_percent": {k: 100 * v for k, v in credits.items()},
            "efficiency_HHV": {"engine": r["efficiency"]["HHV"]["efficiency"], "source": 0.8427,
                               "difference_points": 100 * (r["efficiency"]["HHV"]["efficiency"] - 0.8427)},
            "excess_air": {"engine": r["air"]["excess_air_fraction"], "source": 0.544},
            "stoichiometric_air_kg_kg": {"engine": r["air"]["stoichiometric_dry_air_kg_per_kg_fuel"], "source": 13.92},
            "reading": ("La fuente usa calores específicos constantes (0.23 y 0.45 kcal/kg °C), 584 kcal/kg de calor latente, "
                        "la fórmula O2/(21 − O2) y referencia a la temperatura ambiente; el motor usa entalpías NASA, el "
                        "balance exacto de O2 y referencia a 25 °C con créditos.")}


def main() -> dict:
    out = {"read": "2026-11-14", "not_in_source": NOT_IN_SOURCE, "A_unam_mexico": case_a(), "B_tucuman_argentina": case_b(),
           "C_bee_ejemplo": case_c()}
    (HERE / "33_calderas_reales_publicadas.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    a, b, c = out["A_unam_mexico"], out["B_tucuman_argentina"], out["C_bee_ejemplo"]
    ca = a["comparison"]
    print("A. UNAM, México (medido)")
    print(f"   eficiencia de combustión PCS: motor {100 * ca['combustion_efficiency_HHV']['engine']:.2f} % "
          f"(intervalo {100 * ca['combustion_efficiency_HHV']['engine_range'][0]:.2f}-"
          f"{100 * ca['combustion_efficiency_HHV']['engine_range'][1]:.2f}), analizador 82.26 %, "
          f"{ca['combustion_efficiency_HHV']['difference_points']:+.2f} puntos")
    print(f"   exceso de aire: motor {100 * ca['excess_air']['engine']:.2f} %, analizador 13.18 %; CO2 seco: motor "
          f"{100 * ca['CO2_dry']['engine']:.2f} %, medido 10.28 %")
    dm, dg = a["direct_method"], a["diagnostic"]
    print(f"   método directo: fuente 80.68 %, rehecho {100 * dm['with_source_enthalpies']:.2f} % (sus entalpías), "
          f"{100 * dm['with_iapws']:.2f} % (IAPWS-95)")
    print(f"   diagnóstico completo: por pérdidas {100 * dg['boiler_efficiency_by_losses']:.2f} % (paredes supuestas "
          f"{100 * dg['setting_loss_used']:.2f} % a {100 * dg['load_fraction']:.0f} % de carga), entradas y salidas "
          f"{100 * dg['input_output_efficiency']:.2f} %, diferencia {dg['difference_points']:+.2f} puntos, avisos {dg['data_flags']}")
    cb = b["comparison"]
    print("B. Tucumán, Argentina (medido)")
    print(f"   CO2 seco: motor {100 * cb['CO2_dry']['engine']:.2f} %, medido 11.01 %; lambda: motor {cb['lambda']['engine']:.3f}, "
          "fuente 1.08")
    print(f"   eficiencia directa PCI: motor {100 * cb['direct_efficiency_LHV']['engine']:.2f} %, fuente 74.4 %; por pérdidas sin "
          f"radiación {100 * cb['efficiency_by_losses_LHV_without_radiation']['engine']:.2f} % "
          f"({cb['losses_minus_direct_points']:+.1f} puntos sobre la directa)")
    print("C. Ejemplo de la BEE (resuelto, no es una caldera identificada)")
    for k, v in c["losses_percent"].items():
        print(f"   {k}: motor {v['engine']:.3f}, fuente {v['source']:.3f} ({v['difference_points']:+.3f})")
    print(f"   eficiencia PCS: motor {100 * c['efficiency_HHV']['engine']:.2f} %, fuente 84.27 % "
          f"({c['efficiency_HHV']['difference_points']:+.2f} puntos); créditos {c['credits_percent']}")
    return out


if __name__ == "__main__":
    main()
