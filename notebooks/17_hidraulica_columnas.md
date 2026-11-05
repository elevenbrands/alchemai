# 17 · Packed-column hydraulics and the first fully sized absorber (bloque 23, 2026-11-05)

**What was added.**
- `packages/transport/column_hydraulics.py` computes flooding, irrigated pressure drop and diameter at a fraction of
  flooding, using the Stichlmair, Bravo and Fair (1989) model as implemented in `fluids`.
- **Packing catalog.** Six packings from a secondary source: Arachchige and Melaaen 2012 as reproduced in CET 70,
  AIDIC 2018, Table 2. The PDF table came out jumbled and was rebuilt by columns. One cross-check: Mellapak 250Y gives
  C1 = 1, C2 = 1, C3 = 0.32. Before a final design, confirm with the vendor or the 1989 paper. Any packing can be
  passed as constants.
- **Tools.** `packing_catalog` and `packed_column_hydraulics` (35 tools). They read flows, densities and gas
  viscosity from the DWSIM streams.

## A · Paper example (through `fluids`)

Flooding is 0.63943 m/s and wet ΔP is 539.88 Pa/m. Both match the example in Stichlmair et al. (1989) exactly.

## B · Propane absorber from notebook 16 at 70 % of flooding

| Packing | Diameter (m) | Gas velocity (m/s) | Flooding velocity (m/s) | F-factor (Pa^0.5) | Wet ΔP (Pa/m) |
|---|---|---|---|---|---|
| BX | 1.041 | 0.287 | 0.410 | 0.748 | 257 |
| Flexipac 250Y | 0.968 | 0.332 | 0.474 | 0.866 | 286 |
| Mellapak 250Y | 0.838 | 0.442 | 0.632 | 1.153 | 271 |
| Pall rings 16 mm | 1.200 | 0.216 | 0.308 | 0.563 | 304 |
| Pall rings 25 mm | 1.053 | 0.280 | 0.400 | 0.730 | 305 |
| IMTP 25 | 0.963 | 0.335 | 0.479 | 0.875 | 291 |

The order is the physically expected one. Mellapak 250Y has the highest capacity and the narrowest column, and
16 mm Pall rings the lowest.

## C · First absorber sized end to end (25 mm Pall rings)

Hydraulics give the diameter. Onda and Colburn give the packed height for 94 % recovery. Stichlmair gives the total
ΔP. Seider via IDAES gives the shell cost.

| Diameter | Packed height | H_OG | N_OG | HETP | Total ΔP | Shell (packed height + 3 m) | Shell purchase cost, 2023 USD |
|---|---|---|---|---|---|---|---|
| 1.05 m | 5.22 m | 0.68 m | 7.62 | 0.75 m | 1.59 kPa | 8.2 m | 107,621 |

The diameter changed from the 1 m assumed in notebook 16 to 1.05 m. With it, H_OG drops from 1.15 m to 0.68 m and
the height from 8.8 m to 5.2 m. The difference comes from **setting the packing itself**: a_t = 205 m²/m³ instead of
the illustrative 105, and 25 mm instead of 50 mm. The 25 mm Pall ring geometry comes from the catalog; σc and the
diffusivities are still illustrative (MOCKS.md).

## Limits
- The **packing cost and the internals** (distributors, supports) are not included: only the shell.
- Packing constants come from a secondary source (see above).
- Stichlmair does not cover very viscous liquids or foaming systems. Tray columns (Fair) are not covered.

## The agent on the new card (Opus, `claude -p`, 35 tools)

Card `diametro_columna_relleno`: **1.00 in 61 s** with 9 calls. The agent answered D 0.8384 m, flooding at 0.6317 m/s and
271.13 Pa/m. It solved the streams in DWSIM and passed the statement's constants to `packed_column_hydraulics`. Trace:
`notebooks/bench_runs/v3_hidraulica_cc_opus_run1.mcp_trace.jsonl`.
