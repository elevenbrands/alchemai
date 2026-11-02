# BENCHMARK — evidencia de validación

Cada fila = un resultado reproducible con su comando. Sin fila, no hay ✅.

| Fecha | Qué | Referencia | Resultado | Desviación | Cómo reproducir |
|---|---|---|---|---|---|
| 2026-11-02 | Flash PT C1/C2/C3 (0.7/0.2/0.1) 230 K, 30 bar, PR | `thermo` FlashVL | VF = 0.7426, VL | n/a (smoke) | `uv run python notebooks/smoke_stack.py` |
| 2026-11-02 | Batch adiabático CH4/aire GRI30, 1000 K, 1 atm, 1 s | Cantera 3.2 | T = 1016.4 K | n/a (smoke) | ídem |
| 2026-11-02 | Mezclador agua 100 kg/s 300 K + 50 kg/s 348 K, IAPWS-IF97 | DWSIM 10.2.12 headless (Fluent API, pythonnet) | T = 316.010 K, 150.000 kg/s | n/a (ejemplo oficial) | `uv run python packages/dwsim_client/smoke_fluent.py` |
| 2026-11-02 | Flash PT C1/C2/C3 230 K, 30 bar, PR: DWSIM vs thermo | thermo 0.7426 | DWSIM 0.7456 | 0.4 % (bases de datos de Tc/Pc/kij distintas; ambos PR) | ídem |
