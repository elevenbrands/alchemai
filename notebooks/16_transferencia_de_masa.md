# 16 · Transferencia de masa: absorción por etapas y columna de relleno (bloque 22, 2026-11-05)

Con este bloque, fenómenos de transporte queda cubierto en sus tres partes: momento y calor (bloque 18) y masa (este).

**Qué se agregó.**
- `packages/transport/mass_transfer.py` reúne:
  - Kremser para etapas de equilibrio y Colburn para N_OG;
  - área mojada y coeficientes kL y kG de Onda, Takeuchi y Okumoto (1968);
  - altura Z = H_OG · N_OG con H_OG = H_G + λ·H_L;
  - difusividades por Wilke-Chang (líquido diluido) y Fuller-Schettler-Giddings (gas).
- **Fuentes verificadas.** Las constantes de Onda se leyeron del artículo original en J-STAGE: 1.45, 0.75, 0.1, −0.05 y 0.2 para el área mojada; 0.0051, 2/3, −1/2 y 0.4 para kL; 5.23, 0.7, 1/3 y −2 para kG, con 2.00 para rellenos menores de 15 mm. Las formas de Wilke-Chang y Fuller salen de las notas de Clarkson CH330.
- **Datos obligatorios.** Los volúmenes de difusión, el factor de asociación y los datos del relleno son entradas, porque las tablas cambian entre ediciones.
- **Equipo nuevo.** `absorption_column` en `add_unit`: el absorbedor riguroso de DWSIM, que exige al menos 4 etapas.
- **Herramientas nuevas.** `absorption_stages`, `packed_column_height` y `diffusivity_estimate` (33 tools). Leen ρ, μ y σ de las corrientes de DWSIM.

## A · Kremser contra el absorbedor riguroso de DWSIM

Caso: 100 mol/s de gas con 98 % de metano y 2 % de propano, a 300 K y 10 bar, absorbido con n-decano puro. Paquete
Peng-Robinson. K del propano tomado de un flash de DWSIM; L/V promedio de los extremos de la columna.

| Aceite L (mol/s) | Etapas | K propano | A | Fracción absorbida DWSIM | Kremser | Diferencia (puntos) |
|---|---|---|---|---|---|---|
| 30 | 4 | 0.994 | 0.316 | 0.3190 | 0.3134 | -0.57 |
| 30 | 6 | 0.994 | 0.316 | 0.3209 | 0.3153 | -0.56 |
| 30 | 10 | 0.994 | 0.316 | 0.3211 | 0.3155 | -0.56 |
| 60 | 4 | 0.996 | 0.636 | 0.5966 | 0.5938 | -0.28 |
| 60 | 6 | 0.996 | 0.636 | 0.6210 | 0.6204 | -0.06 |
| 60 | 10 | 0.996 | 0.637 | 0.6328 | 0.6341 | +0.13 |
| 120 | 4 | 0.995 | 1.294 | 0.8888 | 0.8881 | -0.07 |
| 120 | 6 | 0.995 | 1.295 | 0.9419 | 0.9423 | +0.04 |
| 120 | 10 | 0.995 | 1.296 | 0.9815 | 0.9819 | +0.04 |

En 9 casos la diferencia máxima es de 0.6 puntos de fracción absorbida, con A entre 0.32 y 1.30. Con A < 1 la
recuperación se estanca cerca de A, como predice Kremser: 32 % con 30 mol/s de aceite, aunque se agreguen etapas.

## B · Altura de relleno, con datos de relleno ilustrativos

Caso: el mismo absorbedor con 120 mol/s de aceite y 94 % de recuperación, columna de 1 m.

**Datos ilustrativos** (anotados en MOCKS.md): relleno metálico de 50 mm con a_t = 105 m²/m³ y σc = 0.075 N/m.
Las difusividades salen de Fuller y Wilke-Chang con volúmenes de ejemplo.

| H_G | H_L | H_OG | N_OG | Altura | HETP | Resistencia del gas |
|---|---|---|---|---|---|---|
| 0.55 m | 0.72 m | 1.15 m | 7.6 | 8.8 m | 1.26 m | 48 % |

Kremser con los flujos de **entrada** pide 6.95 etapas. La columna rigurosa logra el 94 % con 6 etapas, porque el
metano absorbido aumenta el flujo de líquido y sube A. La herramienta usa los flujos de entrada; para diseño conviene
comparar contra `absorption_column`. La tarjeta del bench exige justamente la columna rigurosa.

## Límites
- **Solo sistemas diluidos.** Kremser y Colburn suponen líneas rectas: si el soluto pasa de unos pocos por ciento, hace falta N_OG numérico con líneas curvas.
- **Sin hidráulica.** No hay inundación, caída de presión ni mojado mínimo. `fluids` trae Stichlmair y Robbins, pero no están conectados todavía.
- **Precisión de Onda.** El artículo reporta ±20 % para el área mojada y kL. No hay validación numérica contra un caso experimental en este bloque.
- **Datos de relleno.** La tensión crítica del material y los datos del relleno no vienen de una tabla verificada.

## The agent on the new card (Opus, `claude -p`, 33 tools)

Card `absorbedor_etapas_minimas`: **1.00 in 91 s**. The agent built the rigorous `absorption_column` with 4, 5, 6 and 7
stages and got 0.8888, 0.9205, 0.9419 and 0.9569. It answered with the minimum, 6 stages at 94.19 %. It made 36 tool
calls. Trace: `notebooks/bench_runs/v3_absorbedor_cc_opus_run1.mcp_trace.jsonl`.
