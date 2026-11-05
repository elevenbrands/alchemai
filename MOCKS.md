# MOCKS — mocks, stubs, placeholders y datos simulados

| Estado | Id | Dónde | Qué | Salida |
|---|---|---|---|---|
| 🟠 | RELLENO-ILUSTRATIVO | `notebooks/16_transferencia_de_masa.py` parte B | Relleno metálico de 50 mm con a_t = 105 m²/m³ y σc = 0.075 N/m, volúmenes de Fuller C 16.5/H 1.98 (tabla de 1966 vía notas de curso), V_A del propano 74.5 cm³/mol: valores de ejemplo sin verificar contra hoja del fabricante ni tabla del libro | La tool `packed_column_height` exige estos datos como entrada; solo el notebook usa estos |
| 🟠 | PRECIOS-ILUSTRATIVOS | `packages/costing/heat_recovery.py` (`ILLUSTRATIVE_PRICES`, `U`), tarjeta `optimo_recuperacion_calor`, notebook 14 | Vapor 8 USD/GJ, agua de enfriamiento 0.5 USD/GJ, electricidad 0.10 USD/kWh y U de 850/1100 W/m²K son valores de ejemplo, no cotizados | Solo demostración del método; `flowsheet_cost` exige los precios del usuario y no tiene valores por defecto |
| ✅ | NOMBRE-PROVISIONAL | repo, `pyproject.toml`, `APP.md` | «proceso-ia» era placeholder | Cerrado 2026-11-02: **AlchemAI**, repo `elevenbrands/alchemai` |

🔴 = entra en una build/entrega y bloquea. 🟠 = pendiente, no bloquea. ✅ = resuelto.
