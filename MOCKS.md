# MOCKS — mocks, stubs, placeholders y datos simulados

| Estado | Id | Dónde | Qué | Salida |
|---|---|---|---|---|
| 🟠 | COLUMNA-DIAMETRO-ILUSTRATIVO | `notebooks/20_variables_enteras.py` (`COST`), tarjeta `optimo_etapas_reflujo` | Diámetro de 1.2 m, espaciamiento de platos 0.6 m y 3 m extra de altura son valores de ejemplo (no dimensionados por hidráulica de platos); precios = PRECIOS-ILUSTRATIVOS | Solo demostración; `flowsheet_cost` exige el diámetro como entrada del usuario |
| 🟠 | PRECIOS-ABSORBEDOR | `notebooks/18_optimizacion_multivariable.py` parte C (`ILLUSTRATIVE`) | Propano 0.5 USD/kg, circulación de solvente 0.0005 USD/kg, refrigeración 6/10/15 USD/GJ: valores de ejemplo | Solo demostración; el resultado cambia de esquina con el precio de refrigeración |
| 🟠 | RELLENO-ILUSTRATIVO | `notebooks/16_transferencia_de_masa.py` parte B | Relleno metálico de 50 mm con a_t = 105 m²/m³ y σc = 0.075 N/m, volúmenes de Fuller C 16.5/H 1.98 (tabla de 1966 vía notas de curso), V_A del propano 74.5 cm³/mol: valores de ejemplo sin verificar contra hoja del fabricante ni tabla del libro | La tool `packed_column_height` exige estos datos como entrada; solo el notebook usa estos |
| 🟠 | PRECIOS-ILUSTRATIVOS | `packages/costing/heat_recovery.py` (`ILLUSTRATIVE_PRICES`, `U`), tarjeta `optimo_recuperacion_calor`, notebook 14 | Vapor 8 USD/GJ, agua de enfriamiento 0.5 USD/GJ, electricidad 0.10 USD/kWh y U de 850/1100 W/m²K son valores de ejemplo, no cotizados | Solo demostración del método; `flowsheet_cost` exige los precios del usuario y no tiene valores por defecto |
| ✅ | NOMBRE-PROVISIONAL | repo, `pyproject.toml`, `APP.md` | «proceso-ia» era placeholder | Cerrado 2026-11-02: **AlchemAI**, repo `elevenbrands/alchemai` |

🔴 = entra en una build/entrega y bloquea. 🟠 = pendiente, no bloquea. ✅ = resuelto.
