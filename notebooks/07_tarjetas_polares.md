# Bloque 13 — Tarjetas polares (bench v2): el paquete termodinámico lo elige el agente

Cuatro tarjetas nuevas en `packages/bench/cards/` (versión `v2`, etiqueta `polares`). Ninguna nombra el
paquete: el enunciado pide «elige tú el paquete termodinámico adecuado y nómbralo en las notas». Las
respuestas esperadas son DWSIM 10.2.12 con su NRTL (los mismos parámetros que FluxoEngine reproduce a 0.1 K,
notebooks 05 y 06). Con Peng-Robinson las cuatro salen mal, así que los números solos califican la elección.

| Tarjeta | Cantidades | DWSIM-NRTL (esperado) | DWSIM-PR | Tolerancia |
|---|---|---|---|---|
| `polar_burbuja_rocio_etanol_agua` — etanol-agua 50/50, 1 atm | T_burbuja, T_rocío | 352.97 K / 357.56 K | 346.39 K / 356.47 K (−6.6 K / −1.1 K) | ±0.5 K |
| `polar_flash_metanol_agua` — 40/60, 355 K, 1 atm | VF, y_MeOH, x_MeOH | 0.5367 / 0.5772 / 0.1946 | 0.5032 / 0.5964 / 0.2010 | ±0.02 / ±0.01 / ±0.01 |
| `polar_columna_metanol_agua` — 20 etapas, alim. 10, R 1.5, B 60 | x_D, x_B, Qc, Qr | 0.9925 / 0.0050 / 3548.2 kW / 3687.1 kW | **no converge** | ±0.01 / ±0.01 / 3 % / 3 % |
| `polar_azeotropo_etanol_agua` — mínimo de la curva de burbuja | x_az, T_az | 0.868 / 351.53 K | 0.700 / 346.58 K (sin mínimo interior: PR no tiene azeótropo) | ±0.02 / ±0.5 K |

Notas:

- La Tb del etanol en DWSIM es 351.8 K (experimental 351.44), por eso el azeótropo de DWSIM queda en 351.53 K
  y x 0.868 (con `thermo` + ChemSep salía 0.876 / 351.33 K, notebook 02; literatura 0.894 / 351.3 K).
- La curva de burbuja es muy plana cerca del azeótropo (≈0.02 K entre x 0.86 y 0.88), por eso la tolerancia en x es 0.02.
- El flash metanol-agua con PR *converge* y queda cerca (ΔVF 0.033, Δy 0.019): la tolerancia se fijó para que PR falle
  sin exigir más precisión de la que el flash de DWSIM da (≈2e-3 en P).
- `sweep` ahora acepta `vary_param="mole_fraction.<Compuesto>"` (los demás compuestos se reparten el resto) y devuelve
  las filas con el mínimo y el máximo del readout: así el agente localiza el azeótropo en una llamada.

## Corrida del agente (claude-code, Opus, 18 tools, sin regla de paquete)

| Tarjeta | Puntaje | s | Paquete elegido | Nota del agente |
|---|---|---|---|---|
| polar_azeotropo_etanol_agua | 1.00 | 45 | NRTL | sweep x 0.70–0.98, luego 0.862–0.878 en pasos de 0.002; mínimo 0.868 / 351.53 K |
| polar_burbuja_rocio_etanol_agua | 1.00 | 25 | NRTL | «polar, non-ideal, azeotrope-forming» |
| polar_columna_metanol_agua | 1.00 | 41 | NRTL | residuos masa 7e-7, energía 1e-12 |
| polar_flash_metanol_agua | 1.00 | 28 | NRTL | V 53.67 / L 46.33 mol/s |

**100 % en 139 s.** No se añadió ninguna regla de selección de paquete al prompt: el modelo elige actividad para agua + alcohol
por su cuenta. Si un modelo más chico (sonnet) falla aquí, esa será la regla 10. Corrida: `notebooks/bench_runs/v2_polares_cc_opus_run1.jsonl`.

Reproducir: `uv run python packages/bench/polar_cases.py` (oráculos NRTL vs PR) ·
`uv run python packages/bench/build_cards.py` (regenera las 39 tarjetas) · `uv run pytest -q tests/test_bench_v2.py`.
