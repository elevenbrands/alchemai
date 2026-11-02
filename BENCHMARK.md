# BENCHMARK — evidencia de validación

Cada fila = un resultado reproducible con su comando. Sin fila, no hay ✅.

| Fecha | Qué | Referencia | Resultado | Desviación | Cómo reproducir |
|---|---|---|---|---|---|
| 2026-11-02 | Flash PT C1/C2/C3 (0.7/0.2/0.1) 230 K, 30 bar, PR | `thermo` FlashVL | VF = 0.7426, VL | n/a (smoke) | `uv run python notebooks/smoke_stack.py` |
| 2026-11-02 | Batch adiabático CH4/aire GRI30, 1000 K, 1 atm, 1 s | Cantera 3.2 | T = 1016.4 K | n/a (smoke) | ídem |
