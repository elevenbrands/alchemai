"""M26 whole-plant case: biodiesel from castor (higuerilla) oil in Sinaloa, one sentence → plant → economics → report.

Run: PYTHONPATH=. uv run python notebooks/26_planta_biodiesel_higuerilla.py
Process: `biodiesel_plant` (BioSTEAM/biorefineries transesterification; castor = declared pseudo-triglyceride).
Economics: `plant_economics_from_model` (the single route: the model's physics + Mexico's economics) with every price and design-basis value it is not given taken from the sourced bank (flagged).
Physical inputs below are the case's DESIGN CHOICES, not data of a real plant: 10 t/h of oil, 0.5 % FFA, 0.05 % water.
"""
import json
from pathlib import Path

from packages.dwsim_client import tools_plant_economics as PE
from packages.dwsim_client import tools_plants as P

OIL_KG_H, HOURS = 10_000.0, 8_410.0           # design choice; 8 410 h/y = Humbird 2011 on-stream convention
OUT = Path(__file__).resolve().parents[1] / "docs" / "plantas"

plant = P.biodiesel_plant(oil_feed_kg_h=OIL_KG_H, oil="castor", ffa_mass_fraction=0.005, water_mass_fraction=0.0005,
                          operating_hours_y=HOURS)
eco = PE.plant_economics_from_model(plant, region="MX-SIN", currency="MXN")
rep = PE.plant_report(eco, "Planta de biodiésel de aceite de higuerilla (Sinaloa) — primer cuadro", 
                      str(OUT / "biodiesel_higuerilla_sinaloa.md"))
s = eco["capital_summary"]
print(json.dumps({"process_verified": plant["evidence"]["verified"], "economics_verified": eco["evidence"]["verified"],
                  "TCI_usd": s.get("TCI_usd") or s, "npv_usd": eco["npv_usd"], "irr": eco["irr"],
                  "min_price": eco["minimum_selling_price"], "assumed": len(eco["assumed_inputs"]),
                  "missing": eco["missing"], "confirm_first": eco["sensitivity"].get("confirm_first"),
                  "report": rep["path"]}, default=float, ensure_ascii=False, indent=1)[:3000])
