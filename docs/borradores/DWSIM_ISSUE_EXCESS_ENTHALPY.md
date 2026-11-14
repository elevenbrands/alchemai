# Borrador — issue para DanWBR/dwsim10

> Estado: BORRADOR, sin publicar. Publicar solo con OK del owner.
> Destino: https://github.com/DanWBR/dwsim10/issues (plantilla «Bug report»).
> Verificado el 2026-11-13: fórmula leída en el código fuente (`master` 9ab62b6, v10.2.16-17) y números medidos en
> DWSIM 10.2.12 con `hex_repro` (12 estados, coincidencia a 0.1 J/mol). No hay issue previo sobre esto.
> Por confirmar antes de publicar: el campo «How did you install it?» y si se repite la medición en 10.2.16.

---

**Title:** `[Bug] Excess enthalpy of the activity-coefficient models uses dγ/dT instead of d ln γ/dT (NRTL, UNIQUAC, UNIFAC, Modified UNIFAC, NIST-MFAC, Wilson)`

### Edition

Cross-platform (Avalonia UI)

### Operating system

macOS

### Architecture

arm64 (Apple Silicon / ARM)

### DWSIM version

10.2.12

### How did you install it?

(por confirmar)

### Describe the bug

The excess enthalpy that the activity-coefficient models add to the liquid enthalpy is computed from the temperature derivative of γ. The Gibbs-Helmholtz relation needs the derivative of ln γ:

```
H^E = −R·T²·Σ x_i·(∂ ln γ_i/∂T)        (P, x constant)
```

DWSIM computes `−R·T²·Σ x_i·(∂γ_i/∂T)`. Each term is therefore multiplied by γ_i. The result is close to right when γ ≈ 1 and too large when the mixture is strongly non-ideal: for ethanol–water with DWSIM's own NRTL parameters it is 1.4 to 4.6 times the Gibbs-Helmholtz value.

It only affects the liquid enthalpy modes that include the excess term, `Excess` and `LiqCp_Excess` ("Experimental Liquid + Excess"). The default mode of these packages, `ExpData`, is not affected. The excess heat capacity (`CPEX_MIX`) is the temperature derivative of the same quantity, so it inherits the error.

**Where** (line numbers of `master` at 9ab62b6; the files are the same in the 10.2.12 build I measured):

`engine/DWSIM.Thermodynamics/PropertyPackages/Models/NRTL.vb`, `DLNGAMMA_DT` (L546) and `HEX_MIX` (L565):

```vb
gamma1 = GAMMA_MR(T, Vx, Vids)
gamma2 = GAMMA_MR(T + epsilon, Vx, Vids)
...
dgamma(i) = (gamma2(i) - gamma1(i)) / (epsilon)      ' L558: dγ/dT, not d ln γ/dT
...
hex += -8.314 * T ^ 2 * Vx(i) * dgamma(i)            ' L572
```

`GAMMA_MR` returns γ, so despite its name `DLNGAMMA_DT` returns dγ/dT. The same pattern is in:

| File (same folder) | dγ/dT line | H^E line |
|---|---|---|
| `NRTL.vb` | 558 | 572 |
| `UNIQUAC.vb` | 596 | 610 |
| `UNIFAC.vb` | 564 | 578 |
| `MODFAC.vb` | 631 | 645 |
| `NISTMFAC.vb` | 647 | 661 |
| `WilsonModel.vb` | 208 (`CalcExcessEnthalpy`) | 211 |

I measured NRTL only. The other five are from reading the code.

**Numbers.** Ethanol (1) + water (2), NRTL with the default parameters DWSIM loads (A12 = −57.9601, A21 = 1241.7396 cal/mol, α = 0.2937), liquid at 3 bar. "DWSIM" is the molar enthalpy in `LiqCp_Excess` minus the molar enthalpy in `ExpData` for the same stream, which isolates the excess term. The two formula columns are computed outside DWSIM from the same parameters.

| T, K | x1 | γ1 | γ2 | DWSIM, J/mol | −RT²Σx·∂γ/∂T | −RT²Σx·∂ln γ/∂T | ratio |
|---|---|---|---|---|---|---|---|
| 298.15 | 0.05 | 5.205 | 1.009 | 971.4 | 971.4 | 211.6 | 4.59 |
| 298.15 | 0.10 | 3.899 | 1.032 | 1125.0 | 1125.0 | 362.7 | 3.10 |
| 298.15 | 0.30 | 1.828 | 1.237 | 825.7 | 825.7 | 579.5 | 1.42 |
| 298.15 | 0.50 | 1.263 | 1.572 | 751.4 | 751.4 | 484.7 | 1.55 |
| 298.15 | 0.70 | 1.068 | 2.010 | 597.2 | 597.2 | 290.2 | 2.06 |
| 298.15 | 0.90 | 1.006 | 2.530 | 232.3 | 232.3 | 89.1 | 2.61 |
| 350.00 | 0.05 | 4.174 | 1.007 | 815.9 | 815.9 | 216.3 | 3.77 |
| 350.00 | 0.10 | 3.309 | 1.026 | 1043.1 | 1043.1 | 378.4 | 2.76 |
| 350.00 | 0.30 | 1.750 | 1.196 | 922.3 | 922.3 | 652.3 | 1.41 |
| 350.00 | 0.50 | 1.254 | 1.485 | 848.5 | 848.5 | 585.6 | 1.45 |
| 350.00 | 0.70 | 1.068 | 1.879 | 699.9 | 699.9 | 376.4 | 1.86 |
| 350.00 | 0.90 | 1.006 | 2.369 | 297.8 | 297.8 | 124.7 | 2.39 |

DWSIM matches the dγ/dT expression to 0.1 J/mol in all 12 states. The d ln γ/dT column agrees with the analytical H^E of an independent NRTL implementation (`thermo.nrtl.NRTL.HE`, same parameters): 362.7 J/mol at 298.15 K and x1 = 0.10, 378.4 at 350 K and x1 = 0.10, 484.7 at 298.15 K and x1 = 0.50.

**Effect on a flowsheet.** In an ethanol–water column (10 mol % ethanol feed at 350 K, 20 trays, reflux ratio 0.70, `LiqCp_Excess`, NRTL parameters regressed from the VLE data of Kamihama et al. 2012 and written into the package) this term changes Q_reboiler − Q_condenser by about 56 kW on a reboiler duty of about 1 MW. Most of it comes from the feed enthalpy (1083 J/mol in DWSIM against 388 J/mol from Gibbs-Helmholtz). With methanol–water, where γ stays near 1, the same comparison agrees within 0.6 %.

**Suggested fix.** Take the derivative of the logarithm:

```vb
dgamma(i) = (Math.Log(gamma2(i)) - Math.Log(gamma1(i))) / (epsilon)
```

and the same in `WilsonModel.CalcExcessEnthalpy`. For NRTL, `GAMMA_DERIVS` (L425) already returns an analytical `dlngdT` that `HEX_MIX` could use. `CPEX_MIX` needs no change of its own.

### Steps to reproduce

1. New simulation with Ethanol and Water, property package NRTL, default interaction parameters.
2. One material stream: 298.15 K, 3 bar, 10 mol % ethanol.
3. Set the property package's enthalpy calculation mode (`EnthalpyEntropyCpCvCalculationMode`) to `ExpData` and note the molar enthalpy of the stream. Switch it to `LiqCp_Excess` and note it again.
4. The difference is 1125 J/mol. Gibbs-Helmholtz with the same parameters gives 363 J/mol.

The two formula columns of the table come from this script (no DWSIM needed):

```python
import math
R_CAL, R = 1.98721, 8.314
A12, A21, AL = -57.9601, 1241.7396, 0.2937        # cal/mol, cal/mol, -  (ethanol/water, DWSIM default)

def gammas(T, x1):
    x2 = 1 - x1
    t12, t21 = A12 / (R_CAL * T), A21 / (R_CAL * T)
    G12, G21 = math.exp(-AL * t12), math.exp(-AL * t21)
    l1 = x2**2 * (t21 * (G21 / (x1 + x2 * G21))**2 + t12 * G12 / (x2 + x1 * G12)**2)
    l2 = x1**2 * (t12 * (G12 / (x2 + x1 * G12))**2 + t21 * G21 / (x1 + x2 * G21)**2)
    return math.exp(l1), math.exp(l2)

def excess_terms(T, x1, h=1e-3):
    a, b, xs = gammas(T - h, x1), gammas(T + h, x1), (x1, 1 - x1)
    d_gamma = sum(x * (q - p) / (2 * h) for x, p, q in zip(xs, a, b))
    d_ln_gamma = sum(x * (math.log(q) - math.log(p)) / (2 * h) for x, p, q in zip(xs, a, b))
    return -R * T * T * d_gamma, -R * T * T * d_ln_gamma   # what DWSIM adds, Gibbs-Helmholtz

print(excess_terms(298.15, 0.10))   # (1125.0, 362.7) J/mol
```

### Additional context

Found while checking an equation-oriented NRTL column against DWSIM with identical parameters: the duties agreed only when both sides left the excess term out.
