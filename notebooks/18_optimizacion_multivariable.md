# 18 · Multivariable optimization `optimize_design_multi` (bloque 24, 2026-11-05)

**What it does.** It optimizes 2 to 4 design variables together. This matters when the variables interact, as the
intermediate pressures of a multistage compressor do. The method has two steps:
1. A coarse grid picks the starting point.
2. Nelder-Mead refines it in normalized coordinates, with optional log scale.

Constraint violations and unsolved points are penalized, and the reported optimum is always an evaluated feasible
point. The tool flags active constraints and variables sitting on a bound. 36 tools in total.

**Choosing the method.** First I tested scipy's Powell with bounds on a quadratic valley: it used 2000 evaluations
without reaching the minimum. Nelder-Mead with bounds solved it in 99 and was chosen. There is also a new readout,
`component_flow.<Compound>`, giving a compound's molar flow in a stream.

## A and B · Three-stage air compression from 1 to 27 bar (intercooling to 300 K, η = 75 %, PR)

| Case | P after stage 1 | P after stage 2 | Total power | Discharges (K) | Evaluations |
|---|---|---|---|---|---|
| No limit | **3.021 bar** | **9.087 bar** | 891.84 kW | 447.0 / 446.4 / 444.7 | 80 |
| Theory (ideal gas, equal ratios) | 3.000 bar | 9.000 bar | — | — | — |
| Stage 1 ≤ 430 K (HAZOP) | **2.699 bar** (limit active) | **8.591 bar** | 892.68 kW (+0.84) | 430.0 / 455.1 / 453.3 | 145 |

With stage 1 capped, the second pressure rebalances to 8.59 bar. That is close to √(2.699 × 27) = 8.54 bar, which splits
the remaining compression equally.

## C · Propane absorber: oil flow and oil chilling temperature

Annual cost = lost propane (0.5 USD/kg) + solvent circulation (0.0005 USD/kg circulated) + chilling. All prices are
**illustrative** (MOCKS.md). The column has 6 stages, oil runs from 40 to 300 mol/s and chilling from 250 to 299 K.

| Refrigeration price | Optimal strategy | Annual cost |
|---|---|---|
| 6 USD/GJ (range down to 270 K) | Chill to the 270 K bound, 67 mol/s oil | 287 233 USD/y |
| 10 USD/GJ | Chill to the 250 K bound, 43 mol/s oil, 96.6 % recovery | 306 607 USD/y |
| 15 USD/GJ | Do not chill (299 K bound), 122 mol/s oil, 95.1 % recovery | 328 779 USD/y |

**Finding.** The problem has two strategies that compete: little cold oil, or plenty of warm oil. With roughly linear
costs the optimum falls in a corner, and the refrigeration price decides which one. At 10 USD/GJ the other corner, warm
oil, would cost about 323 000 USD/y, and the optimizer chose the cheaper one. The tool flags the variable at its bound
instead of hiding it.

## Limits
- Nelder-Mead is local. The initial grid reduces the risk of a bad local optimum but does not remove it.
- With 3-4 variables the grid drops to 3 points per variable.
- No integer variables (number of stages): they still have to be scanned by hand.

## The agent on the new card (Opus, `claude -p`, 36 tools)

Card `optimo_compresion_tres_etapas`: **1.00 in 227 s**. It built the three stages and both coolers, then made a single call
to `optimize_design_multi` with the HAZOP limit. It answered 2.699 bar, 8.595 bar and 892.68 kW from 26 tool calls in total.
Trace: `notebooks/bench_runs/v3_multivariable_cc_opus_run1.mcp_trace.jsonl`.
