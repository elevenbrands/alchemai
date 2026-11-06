"""Held-out reference processes (bloque 25) as hand-built DEXPI Process files. Same conventions as `cases.py`.

Operating numbers are illustrative design-basis values: the sources give the process and the hazards, and only some
of the conditions. Node identifiers match `heldout.yaml`. DESCRIPTIONS are what the precision reviewer sees.
"""
from __future__ import annotations

from pathlib import Path

from packages.hazop.book.cases import HX, TANK_RX, build_model


def marhavilas_estabilizadora():
    comps = ["Hydrogen sulfide", "Methane", "N-butane", "N-heptane", "N-decane", "Water"]
    feed = [0.22, 0.10, 0.08, 0.25, 0.33, 0.02]
    stab = [0.0001, 0.001, 0.05, 0.35, 0.5989, 0.0]
    gas = [0.55, 0.30, 0.12, 0.03, 0.0, 0.0]
    steam = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    steps = [("SRC-1", "Source", "", {}), ("V-101", "SeparatingByGravity", "Three-phase inlet separator", {}),
             ("E-102", "ExchangingThermalEnergy", "Feed/bottoms exchanger", HX),
             ("V-102", "Distilling", "Crude stabiliser column", {}),
             ("SRC-STM", "Source", "", {}), ("FV-110", "RegulatingFlow", "Steam flow control valve to reboiler", {}),
             ("E-101", "ExchangingThermalEnergy", "Steam-heated reboiler", HX), ("SNK-COND", "Sink", "", {}),
             ("P-101", "Pumping", "Stabilised crude pump (spare installed)", {}),
             ("E-103", "RemovingThermalEnergy", "Seawater cooler", {}), ("SNK-STORAGE", "Sink", "", {}),
             ("V-103", "SeparatingByFlash", "Overhead gas scrubber", {}), ("C-101", "Compressing", "Two-stage reciprocating compressor", {}),
             ("SNK-GAS", "Sink", "", {}), ("SNK-H2O", "Sink", "", {}), ("SNK-LIQ", "Sink", "", {})]
    conns = [("1", "SRC-1", "V-101", 20, 20.0, 85000, feed, 0.15), ("2", "V-101", "E-102", 20, 19.0, 80000, feed, 0.0),
             ("2w", "V-101", "SNK-H2O", 20, 19.0, 3000, [0.001, 0.0, 0.0, 0.0, 0.0, 0.999], 0.0),
             ("3", "E-102", "V-102", 95, 8.0, 80000, feed, 0.2),
             ("4", "V-102", "E-101", 180, 7.5, 60000, stab, 0.0), ("5", "E-101", "V-102", 190, 7.5, 60000, stab, 0.3),
             ("6", "SRC-STM", "FV-110", 215, 20.7, 6000, steam, 1.0), ("7", "FV-110", "E-101", 212, 19.0, 6000, steam, 1.0),
             ("8", "E-101", "SNK-COND", 180, 18.0, 6000, steam, 0.0),
             ("9", "V-102", "P-101", 185, 7.5, 62000, stab, 0.0), ("10", "P-101", "E-102", 186, 12.0, 62000, stab, 0.0),
             ("11", "E-102", "E-103", 110, 11.0, 62000, stab, 0.0), ("12", "E-103", "SNK-STORAGE", 40, 10.0, 62000, stab, 0.0),
             ("13", "V-102", "V-103", 60, 7.0, 18000, gas, 1.0), ("14", "V-103", "C-101", 58, 6.8, 17500, gas, 1.0),
             ("15", "V-103", "SNK-LIQ", 58, 6.8, 500, [0.05, 0.0, 0.3, 0.6, 0.05, 0.0], 0.0),
             ("16", "C-101", "SNK-GAS", 130, 28.0, 17500, gas, 1.0)]
    return build_model("marhavilas_estabilizadora", comps, steps, conns)


def single_hexano():
    comps = ["N-hexane", "Water"]
    hx = [0.999, 0.001]
    steps = [("SRC-TRUCK", "Source", "", {}), ("J-UNLOAD", "Pumping", "Tank-truck unloading pump", {}),
             ("XV-IN", "ShuttingOffFlow", "Tank inlet valve", {}),
             ("T-301", "StoringInTank", "Outdoor hexane storage tank, vapour-pressure padded, with dike", {}),
             ("XV-OUT", "ShuttingOffFlow", "Tank outlet valve", {}), ("J-OUT", "Pumping", "Make-up/transfer pump 4-41", {}),
             ("SNK-PROC", "Sink", "", {})]
    conns = [("1", "SRC-TRUCK", "J-UNLOAD", 20, 1.013, 30000, hx, 0.0), ("2", "J-UNLOAD", "XV-IN", 20, 4.0, 30000, hx, 0.0),
             ("3", "XV-IN", "T-301", 20, 1.2, 30000, hx, 0.0), ("4", "T-301", "XV-OUT", 20, 1.2, 8000, hx, 0.0),
             ("5", "XV-OUT", "J-OUT", 20, 1.1, 8000, hx, 0.0), ("6", "J-OUT", "SNK-PROC", 21, 5.0, 8000, hx, 0.0)]
    return build_model("single_hexano", comps, steps, conns)


def benedetti_nitrato():
    comps = ["Ammonium nitrate", "Water", "Ammonia"]
    an = [0.52, 0.479, 0.001]
    steam = [0.0, 1.0, 0.0]
    nh3 = [0.0, 0.0, 1.0]
    steps = [("SRC-AN", "Source", "", {}), ("SRC-NH3", "Source", "", {}), ("FV-NH3", "RegulatingFlow", "Ammonia feed valve for pH control", {}),
             ("SRC-WASH", "Source", "", {}), ("XV-WASH", "ShuttingOffFlow", "Condensate wash / flood line valve", {}),
             ("T-AN", "StoringInTank", "Hot 83 % ammonium nitrate solution tank, 130-150 C, vented, with steam coil", {}),
             ("SRC-STM", "Source", "", {}), ("PCV-STM", "RegulatingFlow", "Steam pressure control valve", {}),
             ("TCV-STM", "RegulatingFlow", "Steam temperature control valve", {}),
             ("E-COIL", "ExchangingThermalEnergy", "Steam heating coil inside the tank", HX), ("SNK-COND", "Sink", "", {}),
             ("XV-REC", "ShuttingOffFlow", "Recirculation/discharge valve", {}), ("P-AN", "Pumping", "Discharge pump (duty + standby)", {}),
             ("SNK-PLANT", "Sink", "", {}), ("SNK-VENT", "Sink", "", {})]
    conns = [("1", "SRC-AN", "T-AN", 140, 1.5, 40000, an, 0.0),
             ("2", "SRC-NH3", "FV-NH3", 25, 6.0, 50, nh3, 1.0), ("3", "FV-NH3", "T-AN", 25, 1.5, 50, nh3, 1.0),
             ("4", "SRC-WASH", "XV-WASH", 90, 3.0, 0.0, steam, 0.0), ("5", "XV-WASH", "T-AN", 90, 1.5, 0.0, steam, 0.0),
             ("6", "SRC-STM", "PCV-STM", 180, 10.0, 1500, steam, 1.0), ("7", "PCV-STM", "TCV-STM", 165, 7.0, 1500, steam, 1.0),
             ("8", "TCV-STM", "E-COIL", 160, 6.0, 1500, steam, 1.0), ("9", "E-COIL", "SNK-COND", 150, 5.5, 1500, steam, 0.0),
             ("10", "T-AN", "E-COIL", 140, 1.013, 40000, an, 0.0), ("11", "E-COIL", "T-AN", 145, 1.013, 40000, an, 0.0),
             ("12", "T-AN", "XV-REC", 140, 1.013, 40000, an, 0.0), ("13", "XV-REC", "P-AN", 140, 1.0, 40000, an, 0.0),
             ("14", "P-AN", "SNK-PLANT", 140, 4.0, 40000, an, 0.0),
             ("15", "T-AN", "SNK-VENT", 140, 1.013, 100, [0.0, 0.98, 0.02], 1.0)]
    return build_model("benedetti_nitrato", comps, steps, conns)


def wsu_hidrogeno_liquido():
    comps = ["Hydrogen"]
    h2 = [1.0]
    steps = [("SRC-DEWAR", "Source", "", {}), ("XV-FILL", "ShuttingOffFlow", "LH2 fill quick-connect", {}),
             ("V-H2", "StoringInPressureVessel", "Vapour-cooled-shield liquid hydrogen tank, MAWP 45 psig, dual rupture discs", {}),
             ("HTR-1", "HeatingElectrical", "Electric boil-off heater, ~10 W, pressure control loop", {}),
             ("PCV-FC", "RegulatingFlow", "Pressure regulator to fuel cell, 20 psig", {}), ("SNK-FC", "Sink", "", {}),
             ("SNK-VENT", "Sink", "", {})]
    conns = [("1", "SRC-DEWAR", "XV-FILL", -253, 1.5, 2.0, h2, 0.0), ("2", "XV-FILL", "V-H2", -253, 1.4, 2.0, h2, 0.0),
             ("3", "V-H2", "HTR-1", -253, 2.4, 0.3, h2, 0.0), ("4", "HTR-1", "V-H2", -245, 2.4, 0.3, h2, 1.0),
             ("5", "V-H2", "PCV-FC", -240, 2.4, 0.3, h2, 1.0), ("6", "PCV-FC", "SNK-FC", 20, 1.3, 0.3, h2, 1.0),
             ("7", "V-H2", "SNK-VENT", -240, 2.4, 0.0, h2, 1.0)]
    return build_model("wsu_hidrogeno_liquido", comps, steps, conns)


def michigan_amoniaco_deshielo():
    comps = ["Ammonia"]
    nh3 = [1.0]
    steps = [("SRC-HG", "Source", "", {}), ("XV-HG", "RegulatingFlow", "Hot-gas main and soft-gas valves (open in defrost)", {}),
             ("SRC-LIQ", "Source", "", {}), ("XV-LIQ", "ShuttingOffFlow", "Liquid ammonia feed valve with downstream check valve (closed in defrost)", {}),
             ("E-EVAP", "ExchangingThermalEnergy", "Evaporator coil in defrost mode (hot gas inside, frost outside)", HX),
             ("XV-SUC", "ShuttingOffFlow", "Suction stop valve to low-temperature suction line (closed in defrost)", {}),
             ("SNK-COMP", "Sink", "", {}), ("XV-RELIEF", "RegulatingFlow", "Defrost pressure-relief regulator", {}), ("SNK-RELIEF", "Sink", "", {})]
    conns = [("1", "SRC-HG", "XV-HG", 70, 10.0, 600, nh3, 1.0), ("2", "XV-HG", "E-EVAP", 65, 7.0, 600, nh3, 1.0),
             ("3", "SRC-LIQ", "XV-LIQ", -30, 2.0, 0.0, nh3, 0.0), ("4", "XV-LIQ", "E-EVAP", -30, 1.5, 0.0, nh3, 0.0),
             ("5", "E-EVAP", "XV-SUC", 10, 6.5, 0.0, nh3, 0.5), ("6", "XV-SUC", "SNK-COMP", -30, 1.2, 0.0, nh3, 1.0),
             ("7", "E-EVAP", "XV-RELIEF", 20, 6.5, 600, nh3, 0.3), ("8", "XV-RELIEF", "SNK-RELIEF", 15, 4.0, 600, nh3, 0.4)]
    return build_model("michigan_amoniaco_deshielo", comps, steps, conns)


HELDOUT = {f.__name__: f for f in (marhavilas_estabilizadora, single_hexano, benedetti_nitrato, wsu_hidrogeno_liquido,
                                   michigan_amoniaco_deshielo)}

DESCRIPTIONS = {
    "marhavilas_estabilizadora": "Crudo amargo (22 % mol H2S) de una plataforma entra a 20 bar y 20 °C a un separador trifásico V-101, "
        "pasa por el intercambiador alimentación/fondos E-102 y entra a la estabilizadora V-102 (8 bar). El rehervidor E-101 se calienta "
        "con vapor de 20.7 bar regulado por FV-110. Los fondos (crudo estabilizado, 185 °C) van por la bomba P-101 a E-102 y al enfriador "
        "de agua de mar E-103, y de ahí a almacenamiento a 40 °C. El gas de domo (rico en H2S y metano) va al lavador V-103 y al compresor "
        "reciprocante C-101 que lo manda a 28 bar.",
    "single_hexano": "Tanque exterior de hexano líquido T-301, presurizado solo por su presión de vapor, con dique. Se llena desde una pipa "
        "con la bomba de descarga J-UNLOAD y la válvula XV-IN; la bomba de transferencia J-OUT saca hexano por XV-OUT hacia el proceso. "
        "Ambiente exterior, 20 °C nominal; el hexano trae trazas de agua.",
    "benedetti_nitrato": "Tanque atmosférico ventilado T-AN con ~800 t de solución de nitrato de amonio al 83 % a 130-150 °C, mantenida "
        "caliente por un serpentín de vapor E-COIL (vapor regulado por PCV-STM y TCV-STM). Se dosifica amoniaco gaseoso por FV-NH3 para "
        "control de pH y hay una línea de condensado de lavado/inundación con válvula XV-WASH normalmente cerrada. La bomba P-AN recircula "
        "y descarga la solución a otra planta por XV-REC. El nitrato de amonio se descompone a partir de 260-290 °C y cristaliza si se enfría.",
    "wsu_hidrogeno_liquido": "Tanque criogénico V-H2 de hidrógeno líquido (−253 °C) con escudo enfriado por vapor y aislamiento al vacío; "
        "MAWP 45 psig. Se llena desde un dewar por XV-FILL. Un calentador eléctrico HTR-1 (~10 W) evapora hidrógeno para mantener presión "
        "y el regulador PCV-FC entrega gas a 20 psig a una celda de combustible. Discos de ruptura redundantes al venteo.",
    "michigan_amoniaco_deshielo": "Serpentín de un evaporador de amoniaco E-EVAP en ciclo de deshielo: entra gas caliente de alta presión "
        "(65 °C, 7 bar) por XV-HG; la válvula de líquido XV-LIQ (amoniaco a −30 °C) y la de succión XV-SUC hacia los compresores deben "
        "estar cerradas durante el deshielo, y el regulador de alivio XV-RELIEF saca el gas condensado. La planta almacena decenas de "
        "toneladas de amoniaco anhidro y hay personal cerca.",
}


def write_all(out_dir: str | Path) -> dict[str, str]:
    from packages.hazop.book.cases import write_all as _w
    return _w(out_dir, HELDOUT)
