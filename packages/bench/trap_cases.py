"""Bench v1 "trap" cards: specifications that are physically impossible or internally inconsistent.

A good agent must DETECT the trap (and say why) instead of forcing the simulator or inventing a number. Each
trap lists the keywords (ES/EN, lowercase substrings) that an acceptable explanation must contain, and the
reason the oracle reports. Traps are checked against physics or the simulator, never against taste:
second law (temperature cross), thermodynamic limits (azeotrope, critical point), machine direction (pump,
compressor, valve), stoichiometry (missing oxidant), conservation (mass balance, split ratios) and the FUG
minimum reflux. Keywords are deliberately generous: we grade whether the agent saw the right wall, not its prose.
"""
from __future__ import annotations

SOURCE = "Physics / DWSIM 10.2.12 checks (trap cards are infeasible by construction)"

TRAPS: list[dict] = [
    dict(id="trampa_cruce_temperaturas", tags=["trampa", "intercambiador"],
         title=("Intercambiador con cruce de temperaturas", "Heat exchanger with temperature cross"),
         statement=("Intercambiador a contracorriente, agua-agua (IAPWS-IF97), 1 bar. Lado caliente: 10 kg/s entran a 350 K y deben salir a 300 K. "
                    "Lado frío: 8 kg/s entran a 320 K y deben salir a 360 K. Calcular UA y la carga térmica en kW.",
                    "Counter-current water-water exchanger (IAPWS-IF97), 1 bar. Hot side: 10 kg/s in at 350 K, must leave at 300 K. "
                    "Cold side: 8 kg/s in at 320 K, must leave at 360 K. Compute UA and the duty in kW."),
         quantities={"UA_W_K": "W/K", "Q_kW": "kW"},
         keywords=["cruce", "segunda ley", "cross", "second law", "360", "mayor que la entrada caliente", "hotter than", "above the hot inlet", "exceeds the hot"],
         reason=("Cruce de temperaturas: la salida fría (360 K) supera la entrada caliente (350 K) y la carga de ambos lados no coincide (2.1 vs 1.3 MW). Viola la segunda ley.",
                 "Temperature cross: the cold outlet (360 K) exceeds the hot inlet (350 K) and the two duties do not match (2.1 vs 1.3 MW). Violates the second law.")),
    dict(id="trampa_azeotropo", tags=["trampa", "destilacion", "nrtl"],
         title=("Destilado etanol-agua más allá del azeótropo", "Ethanol-water distillate past the azeotrope"),
         statement=("Columna a 1 atm (NRTL) alimentada con 100 mol/s de etanol-agua 40/60 % mol. Se pide un destilado con 97 % mol de etanol y un fondo con 1 % mol. "
                    "Calcular el flujo de destilado en mol/s y el reflujo mínimo.",
                    "Column at 1 atm (NRTL) fed 100 mol/s of ethanol-water 40/60 mol %. Required: distillate 97 mol % ethanol, bottoms 1 mol %. "
                    "Compute distillate flow in mol/s and the minimum reflux."),
         quantities={"D_mol_s": "mol/s", "R_min": "-"},
         keywords=["azeótropo", "azeotropo", "azeotrope", "89", "0.89", "0.876", "87.6", "pinch"],
         reason=("El azeótropo etanol-agua a 1 atm está en ≈0.89 mol de etanol (0.876 con el NRTL de DWSIM): un destilado de 0.97 por destilación simple es imposible.",
                 "The ethanol-water azeotrope at 1 atm is ≈0.89 mol ethanol (0.876 with DWSIM's NRTL): a 0.97 distillate by simple distillation is impossible.")),
    dict(id="trampa_bomba_descarga", tags=["trampa", "bomba"],
         title=("Bomba con descarga menor que la succión", "Pump with discharge below suction"),
         statement=("Bomba centrífuga, agua (IAPWS-IF97), 20 kg/s a 300 K entra a 5 bar; la presión de descarga especificada es 3 bar; eficiencia 75 %. "
                    "Calcular la potencia al eje en kW y la T de salida.",
                    "Centrifugal pump, water (IAPWS-IF97), 20 kg/s at 300 K enters at 5 bar; specified discharge pressure 3 bar; efficiency 75 %. "
                    "Compute shaft power in kW and outlet T."),
         quantities={"W_kW": "kW", "T_out_K": "K"},
         keywords=["menor", "below", "lower than", "negativ", "no puede", "cannot", "válvula", "valvula", "valve", "expansión", "turbina", "turbine"],
         reason=("Una bomba no reduce la presión: 5 → 3 bar da potencia negativa. Es una caída de presión (válvula), no una bomba.",
                 "A pump cannot lower pressure: 5 → 3 bar gives negative power. This is a pressure drop (valve), not a pump.")),
    dict(id="trampa_reactor_sin_oxidante", tags=["trampa", "reactor"],
         title=("Combustión de metano sin oxígeno", "Methane combustion without oxygen"),
         statement=("Reactor de conversión isotérmico a 298.15 K y 1 atm: CH₄ + 2 O₂ → CO₂ + 2 H₂O con 100 % de conversión de metano. "
                    "Alimentación: 1 mol/s de metano y 9.52 mol/s de nitrógeno puro (sin aire). Calcular el CO₂ producido en mol/s y el calor retirado en kW.",
                    "Isothermal conversion reactor at 298.15 K and 1 atm: CH₄ + 2 O₂ → CO₂ + 2 H₂O with 100 % methane conversion. "
                    "Feed: 1 mol/s methane and 9.52 mol/s pure nitrogen (no air). Compute CO₂ produced in mol/s and heat removed in kW."),
         quantities={"CO2_mol_s": "mol/s", "Q_kW": "kW"},
         keywords=["oxígeno", "oxigeno", "oxygen", "oxidante", "oxidant", "o₂", "o2", "limitante", "limiting", "sin aire", "no air", "negativ"],
         reason=("No hay oxígeno en la alimentación: el reactivo limitante es cero, la conversión de metano es 0 % y no se produce CO₂. Una conversión de 100 % daría flujo negativo de O₂.",
                 "No oxygen in the feed: the limiting reactant is zero, methane conversion is 0 % and no CO₂ forms. Forcing 100 % conversion gives a negative O₂ flow.")),
    dict(id="trampa_compresor_expansion", tags=["trampa", "compresor"],
         title=("Compresor que baja la presión", "Compressor that lowers pressure"),
         statement=("Compresor adiabático, 100 mol/s de gas 85/7/4/2 % C1/C2/C3/nC4 + 2 % CO₂ (Peng-Robinson), entra a 300 K y 10 bar; "
                    "presión de descarga especificada 5 bar; η adiabática 75 %. Calcular la potencia consumida en kW y la T de descarga.",
                    "Adiabatic compressor, 100 mol/s of gas 85/7/4/2 % C1/C2/C3/nC4 + 2 % CO₂ (Peng-Robinson), inlet 300 K and 10 bar; "
                    "specified discharge 5 bar; adiabatic η 75 %. Compute power consumed in kW and discharge T."),
         quantities={"W_kW": "kW", "T_out_K": "K"},
         keywords=["expan", "turbina", "turbine", "menor", "below", "lower", "negativ", "no puede", "cannot", "válvula", "valvula", "valve"],
         reason=("Un compresor no baja la presión: 10 → 5 bar es una expansión (turbina o válvula). La potencia 'consumida' saldría negativa.",
                 "A compressor cannot lower pressure: 10 → 5 bar is an expansion (expander or valve). 'Consumed' power would come out negative.")),
    dict(id="trampa_flash_supercritico", tags=["trampa", "termo"],
         title=("Líquido de metano a 300 K", "Methane liquid at 300 K"),
         statement=("Metano puro (Peng-Robinson), 100 mol/s, a 300 K. Encontrar la presión del separador que deje 30 % mol en fase líquida y reportar la densidad del líquido en kg/m³.",
                    "Pure methane (Peng-Robinson), 100 mol/s, at 300 K. Find the separator pressure that leaves 30 mol % in the liquid phase and report the liquid density in kg/m³."),
         quantities={"P_bar": "bar", "rho_liq_kg_m3": "kg/m3"},
         keywords=["supercrít", "supercrit", "crítica", "critica", "critical", "190", "tc", "no hay líquido", "no liquid", "no condensa", "cannot condense", "single phase", "una sola fase"],
         reason=("Tc del metano = 190.6 K: a 300 K es supercrítico y no existe fase líquida a ninguna presión.",
                 "Methane Tc = 190.6 K: at 300 K it is supercritical and no liquid phase exists at any pressure.")),
    dict(id="trampa_divisor_120", tags=["trampa", "balance"],
         title=("Divisor que reparte 120 %", "Splitter that distributes 120 %"),
         statement=("Divisor de corriente: 100 mol/s de gas 85/7/4/2 % C1/C2/C3/nC4 + 2 % CO₂ a 300 K y 50 bar (Peng-Robinson). Enviar 70 % a la salida A y 50 % a la salida B. "
                    "Calcular el flujo molar de A y de B.",
                    "Stream splitter: 100 mol/s of gas 85/7/4/2 % C1/C2/C3/nC4 + 2 % CO₂ at 300 K and 50 bar (Peng-Robinson). Send 70 % to outlet A and 50 % to outlet B. "
                    "Compute the molar flow of A and B."),
         quantities={"A_mol_s": "mol/s", "B_mol_s": "mol/s"},
         keywords=["120", "1.2", "suma", "sum", "balance", "no suman", "exceed", "más de 100", "more than 100", "mayor que 1", "greater than 1", "fracciones"],
         reason=("Las fracciones de división suman 120 %: viola el balance de masa (saldrían 120 mol/s de 100).",
                 "The split ratios sum to 120 %: mass balance violated (120 mol/s would leave from 100).")),
    dict(id="trampa_mezclador_balance", tags=["trampa", "balance"],
         title=("Mezclador con producto mayor que las entradas", "Mixer with product exceeding the inlets"),
         statement=("Mezclador adiabático: 100 kg/s de agua a 300 K y 50 kg/s de agua a 350 K, ambas a 2 bar (IAPWS-IF97). El producto debe ser de 200 kg/s. "
                    "Calcular la T del producto y el flujo del producto en kg/s.",
                    "Adiabatic mixer: 100 kg/s water at 300 K and 50 kg/s water at 350 K, both at 2 bar (IAPWS-IF97). The product must be 200 kg/s. "
                    "Compute the product T and product flow in kg/s."),
         quantities={"T_out_K": "K", "F_out_kg_s": "kg/s"},
         keywords=["150", "balance", "conserv", "no puede ser 200", "cannot be 200", "suma", "sum", "mass"],
         reason=("Conservación de masa: 100 + 50 = 150 kg/s, no 200. El producto solo puede ser 150 kg/s (a ≈316.6 K).",
                 "Mass conservation: 100 + 50 = 150 kg/s, not 200. The product can only be 150 kg/s (at ≈316.6 K).")),
    dict(id="trampa_valvula_sube_presion", tags=["trampa", "valvula"],
         title=("Válvula que sube la presión", "Valve that raises pressure"),
         statement=("Válvula de expansión isoentálpica: 100 mol/s de gas 85/7/4/2 % C1/C2/C3/nC4 + 2 % CO₂ (Peng-Robinson) a 300 K y 10 bar; presión de salida especificada 20 bar. "
                    "Calcular la T de salida y ΔT.",
                    "Isenthalpic valve: 100 mol/s of gas 85/7/4/2 % C1/C2/C3/nC4 + 2 % CO₂ (Peng-Robinson) at 300 K and 10 bar; specified outlet pressure 20 bar. "
                    "Compute outlet T and ΔT."),
         quantities={"T_out_K": "K", "dT_K": "K"},
         keywords=["válvula", "valvula", "valve", "no puede", "cannot", "subir", "aumentar", "raise", "increase", "compresor", "compressor", "mayor que la entrada", "above the inlet"],
         reason=("Una válvula solo baja la presión: 10 → 20 bar requiere un compresor. La especificación es imposible.",
                 "A valve can only lower pressure: 10 → 20 bar needs a compressor. The specification is impossible.")),
    dict(id="trampa_reflujo_bajo_minimo", tags=["trampa", "destilacion"],
         title=("Depropanizadora con reflujo menor al mínimo", "Depropanizer with reflux below minimum"),
         statement=("Depropanizadora FUG (Peng-Robinson): alimentación líquida saturada 100 mol/s (5/45/40/10 % C2/C3/nC4/nC5) a 17 bar; 2 % nC4 en destilado y 2 % C3 en fondo; "
                    "condensador 16 bar, rehervidor 17.5 bar. Operar con R = 0.6. Calcular el número de etapas N y la carga del condensador en kW.",
                    "FUG depropanizer (Peng-Robinson): saturated-liquid feed 100 mol/s (5/45/40/10 % C2/C3/nC4/nC5) at 17 bar; 2 % nC4 in distillate and 2 % C3 in bottoms; "
                    "condenser 16 bar, reboiler 17.5 bar. Operate at R = 0.6. Compute the number of stages N and the condenser duty in kW."),
         quantities={"N": "-", "Qc_kW": "kW"},
         keywords=["reflujo mínimo", "reflujo minimo", "minimum reflux", "r_min", "rmin", "r min", "1.2", "1.23", "infinit", "pinch", "menor que el mínimo", "below the minimum", "below rmin"],
         reason=("El reflujo mínimo FUG de esta separación es R_min ≈ 1.28 (DWSIM, Underwood); con R = 0.6 < R_min la separación es imposible (etapas infinitas / Gilliland fuera de rango).",
                 "The FUG minimum reflux for this separation is R_min ≈ 1.28 (DWSIM, Underwood); at R = 0.6 < R_min the separation is impossible (infinite stages / Gilliland out of range).")),
]

# ---- v1b: specification traps (under-/over-specified statements). The agent must use dof_check and refuse to invent.
DOF_KW_UNDER = ["under", "sub-especific", "subespecific", "falta", "missing", "grados de libertad", "degrees of freedom", "dof", "no se da", "not given", "omit", "sin especificar", "unspecified"]
DOF_KW_OVER = ["over", "sobre-especific", "sobreespecific", "ambos", "both", "contradic", "inconsist", "conflict", "redundan", "no puede fijarse", "cannot be specified", "cannot both", "dos especificaciones"]
TRAPS += [
    dict(id="trampa_dof_calentador_doble", tags=["trampa", "dof", "over"],
         title=("Calentador con temperatura Y carga fijadas", "Heater with both outlet temperature and duty fixed"),
         statement=("Calentador de agua (IAPWS-IF97): 10 kg/s a 300 K y 2 bar. Especificaciones: temperatura de salida 350 K y carga térmica 500 kW. "
                    "Calcular la temperatura de salida y la carga en kW.",
                    "Water heater (IAPWS-IF97): 10 kg/s at 300 K and 2 bar. Specifications: outlet temperature 350 K and heat duty 500 kW. "
                    "Compute outlet temperature and duty in kW."),
         quantities={"T_out_K": "K", "Q_kW": "kW"},
         keywords=DOF_KW_OVER + ["2090", "2 090", "2.09", "2,09"],
         reason=("Sobre-especificado: un calentador acepta T de salida O carga, no ambas. Además son incompatibles: 10 kg/s × 4.18 kJ/kg·K × 50 K ≈ 2 090 kW, no 500 kW.",
                 "Over-specified: a heater takes outlet T OR duty, not both. They also disagree: 10 kg/s × 4.18 kJ/kg·K × 50 K ≈ 2 090 kW, not 500 kW.")),
    dict(id="trampa_dof_flash_sin_presion", tags=["trampa", "dof", "under"],
         title=("Flash sin presión", "Flash without a pressure"),
         statement=("Separador flash (Peng-Robinson): gas natural 80/8/4/2/2 % C1/C2/C3/nC4/nC5 + 2 % N₂ + 2 % CO₂, 100 mol/s, a 230 K. "
                    "Calcular la fracción vapor y la fracción mol de metano en el vapor.",
                    "Flash separator (Peng-Robinson): natural gas 80/8/4/2/2 % C1/C2/C3/nC4/nC5 + 2 % N₂ + 2 % CO₂, 100 mol/s, at 230 K. "
                    "Compute the vapor fraction and the methane mole fraction in the vapor."),
         quantities={"VF": "-", "y_CH4": "-"},
         keywords=DOF_KW_UNDER + ["presión", "presion", "pressure"],
         reason=("Sub-especificado: falta la presión del flash; la fracción vapor a 230 K va de ~0.6 a 1 según la presión. No hay respuesta sin ese dato.",
                 "Under-specified: the flash pressure is missing; at 230 K the vapor fraction ranges from ~0.6 to 1 with pressure. No answer without it.")),
    dict(id="trampa_dof_compresor_sin_composicion", tags=["trampa", "dof", "under"],
         title=("Compresor sin composición del gas", "Compressor without the gas composition"),
         statement=("Compresor adiabático, η 75 %: 'gas de proceso' 100 mol/s a 300 K de 10 a 30 bar (Peng-Robinson). Calcular potencia en kW y T de descarga.",
                    "Adiabatic compressor, η 75 %: 'process gas' 100 mol/s at 300 K from 10 to 30 bar (Peng-Robinson). Compute power in kW and discharge T."),
         quantities={"W_kW": "kW", "T_out_K": "K"},
         keywords=DOF_KW_UNDER + ["composición", "composicion", "composition", "qué gas", "which gas", "cp/cv", "gamma", "κ"],
         reason=("Sub-especificado: no se da la composición del gas; la potencia y la T de descarga dependen de cp/cv y del peso molecular (H₂ vs CH₄ vs CO₂ difieren en factores de 2 a 10).",
                 "Under-specified: no gas composition; power and discharge T depend on cp/cv and molecular weight (H₂ vs CH₄ vs CO₂ differ by factors of 2 to 10).")),
    dict(id="trampa_dof_columna_sin_reflujo", tags=["trampa", "dof", "under", "destilacion"],
         title=("Columna FUG sin reflujo ni etapas", "FUG column with neither reflux nor stages"),
         statement=("Depropanizadora FUG (Peng-Robinson): alimentación líquida saturada 100 mol/s (5/45/40/10 % C2/C3/nC4/nC5) a 17 bar; 2 % nC4 en destilado y 2 % C3 en fondo; "
                    "condensador 16 bar, rehervidor 17.5 bar. Calcular el número de etapas teóricas N y la carga del rehervidor en kW.",
                    "FUG depropanizer (Peng-Robinson): saturated-liquid feed 100 mol/s (5/45/40/10 % C2/C3/nC4/nC5) at 17 bar; 2 % nC4 in distillate and 2 % C3 in bottoms; "
                    "condenser 16 bar, reboiler 17.5 bar. Compute the theoretical stages N and the reboiler duty in kW."),
         quantities={"N": "-", "Qr_kW": "kW"},
         keywords=DOF_KW_UNDER + ["reflujo", "reflux", "r/rmin", "etapas", "stages"],
         reason=("Sub-especificado: N y la carga del rehervidor dependen del reflujo (o de N dado R); solo N_min y R_min están definidos. Falta R (o R/R_min).",
                 "Under-specified: N and reboiler duty depend on the reflux ratio (or N given R); only N_min and R_min are defined. R (or R/R_min) is missing.")),
    dict(id="trampa_dof_mezclador_t_fijada", tags=["trampa", "dof", "over"],
         title=("Mezclador con temperatura de salida impuesta", "Mixer with an imposed outlet temperature"),
         statement=("Mezclador adiabático (IAPWS-IF97): 100 kg/s de agua a 300 K y 50 kg/s a 350 K, ambas a 2 bar. La corriente de salida debe estar a 340 K. "
                    "Calcular la T de salida y el flujo del producto en kg/s.",
                    "Adiabatic mixer (IAPWS-IF97): 100 kg/s water at 300 K and 50 kg/s at 350 K, both at 2 bar. The outlet stream must be at 340 K. "
                    "Compute the outlet T and the product flow in kg/s."),
         quantities={"T_out_K": "K", "F_out_kg_s": "kg/s"},
         keywords=DOF_KW_OVER + ["316", "balance de energía", "energy balance", "adiab", "calculada", "computed", "resultado", "result"],
         reason=("Sobre-especificado: la T de salida de un mezclador adiabático la fija el balance de energía (≈316.6 K), no puede imponerse en 340 K sin un calentador.",
                 "Over-specified: an adiabatic mixer's outlet T is set by the energy balance (≈316.6 K); it cannot be imposed at 340 K without a heater.")),
]


def trap_card(t: dict) -> dict:
    return {"id": t["id"], "kind": "trap", "version": "v1b" if "dof" in t["tags"] else "v1", "title": {"es": t["title"][0], "en": t["title"][1]},
            "statement": {"es": t["statement"][0], "en": t["statement"][1]}, "tags": t["tags"],
            "expected": {k: {"value": None, "unit": u, "tol": {}} for k, u in t["quantities"].items()},
            "trap": {"keywords": t["keywords"], "reason": {"es": t["reason"][0], "en": t["reason"][1]}},
            "requires": [], "source": SOURCE}
