# BENCHMARK — evidencia de validación

Cada fila = un resultado reproducible con su comando. Sin fila, no hay ✅.

| Fecha | Qué | Referencia | Resultado | Desviación | Cómo reproducir |
|---|---|---|---|---|---|
| 2026-11-02 | Flash PT C1/C2/C3 (0.7/0.2/0.1) 230 K, 30 bar, PR | `thermo` FlashVL | VF = 0.7426, VL | n/a (smoke) | `uv run python notebooks/smoke_stack.py` |
| 2026-11-02 | Batch adiabático CH4/aire GRI30, 1000 K, 1 atm, 1 s | Cantera 3.2 | T = 1016.4 K | n/a (smoke) | ídem |
| 2026-11-02 | Mezclador agua 100 kg/s 300 K + 50 kg/s 348 K, IAPWS-IF97 | DWSIM 10.2.12 headless (Fluent API, pythonnet) | T = 316.010 K, 150.000 kg/s | n/a (ejemplo oficial) | `uv run python packages/dwsim_client/smoke_fluent.py` |
| 2026-11-02 | Flash PT C1/C2/C3 230 K, 30 bar, PR: DWSIM vs thermo | thermo 0.7426 | DWSIM 0.7456 | 0.4 % (bases de datos de Tc/Pc/kij distintas; ambos PR) | ídem |
| 2026-11-02 | Línea base Fluxo vs DWSIM, caso 1: flash PT gas natural 7 comp., 230 K, 30 bar, PR | DWSIM 10.2.12 | VF 0.8751 (Fluxo) vs 0.8762 | +0.13 % VF; mayoritarios < 1.6 %; x_CO₂ +26 % (k_ij) | `uv run python notebooks/baseline_fluxo_vs_dwsim.py` (tabla en `notebooks/baseline_fluxo_vs_dwsim.md`) |
| 2026-11-02 | Caso 2: compresor η_ad 0.75, 50→60 bar, 100 mol/s + enfriador a 300 K | DWSIM 10.2.12 | T 315.8 K, 52.79 kW, Q 79.44 kW (Fluxo) vs 315.8 K, 52.82 kW, 79.34 kW | ≤ 0.13 % ✅ | ídem |
| 2026-11-02 | Caso 3: depropanizadora FUG, 17 bar, LK propano / HK n-butano 2 %, R = 1.84 | DWSIM 10.2.12 (Shortcut Column) | D, Qc, Qr, T_dest, T_fondo ≤ 0.3 % ✅; N_min −6.6 %, R_min −9.5 %, N −16 %, plato alim. 11 vs 8.9 | Fluxo registra 🟠 en su MOCKS.md | ídem |
