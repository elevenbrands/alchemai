"""M26 whole-plant case: ethanol from sugarcane, published reference plant (Huang et al. 2016 via biorefineries) re-priced
for Mexico with the sourced bank. Run: PYTHONPATH=. uv run python notebooks/26_planta_etanol_cana.py

The cane price is deliberately NOT given: Mexico's legal price is per tonne of recoverable standard sugar (LDSCA art. 58)
and needs the mill's KARBE, which only the user has, so the tool must report it as MISSING and still return capital,
costs and the minimum ethanol price as an upper-bound note. Electricity export of the BioSTEAM model is 2-3x the paper's,
so electricity sales are left out here (not passed as a product) to avoid booking a revenue the benchmark contradicts.
"""
import json
from pathlib import Path

from packages.dwsim_client import tools_plant_economics as PE
from packages.dwsim_client import tools_plants as P

OUT = Path(__file__).resolve().parents[1] / "docs" / "plantas"
ref = P.reference_biorefinery("sugarcane_ethanol")
hours = next(d["value"] for d in ref["design_basis"] if d["parameter"] == "horas de operación")
pr = ref["products"]
raw = [{"name": "caña de azúcar", "flow": pr["feed_kg_h"], "unit": "kg/h"}]               # no price: Mexican price needs KARBE
prod = [{"name": "etanol", "flow": pr["ethanol_pure_L_y"] / 1000.0, "unit": "m3/y", "reference_item": "ethanol"}]
eco = PE.plant_economics(equipment=ref["equipment"], raw_materials=raw, products=prod, region="MX", plant_type="ethanol sugar",
                         main_product="etanol", basis={"on_stream_hours": float(hours)})
rep = PE.plant_report(eco, "Planta de etanol de caña (modelo publicado, precios de México) — primer cuadro",
                      str(OUT / "etanol_cana_mexico.md"))
print(json.dumps({"benchmark": [(v["quantity"], round(v["deviation_pct"], 1)) for v in ref["published_benchmark"]["values"]],
                  "hours": hours, "TCI_usd": eco["capital_summary"]["total_capital_investment_usd"],
                  "npv": eco["npv_usd"], "missing": eco["missing"], "min_price": eco["minimum_selling_price"],
                  "assumed": len(eco["assumed_inputs"]), "report": rep["path"]}, default=float, ensure_ascii=False)[:2500])
