# Notebook 03 — Cinética y redes de reactores: libro vs Cantera vs DWSIM

Cantera 3.2.0 · DWSIM 10.2.12 (Fluent API) · fecha 2026-11-02

## A. Primer orden A → B isotérmico (el caso mínimo de Fluxo 1.1)

k = A·exp(−E/RT) con A = 1e+08 1/s, E = 100 kJ/mol, T = 600 K → **k = 0.19697 1/s**. Alimentación 10 mol/s de A a 1 bar; V = 10 m³ → Q_v(gas ideal) = 0.4989 m³/s, **τ = 20.045 s**, kτ = 3.948.

| Reactor | Fórmula | Analítico | Cantera | DWSIM (PR, n-butano→isobutano) | Δ Cantera | Δ DWSIM |
|---|---|---|---|---|---|---|
| PFR | X = 1 − e^(−kτ) | 0.98071 | 0.98071 | 0.98085 | -0.000 % | +0.014 % |
| CSTR | X = kτ/(1+kτ) | 0.79791 | 0.79791 | 0.79813 | +0.000 % | +0.028 % |

DWSIM cierra balances en ambos (verify ok = True); cargas isotérmicas PFR -90.99 kW, CSTR -74.04 kW (isomerización exotérmica, ΔHf del isobutano menor). La desviación de DWSIM (+0.014 % / +0.028 %) es la densidad Peng-Robinson frente al gas ideal (Z ≈ 0.998 a 1 bar y 600 K): τ real ligeramente mayor. Lección: **el reactor cinético de Fluxo 1.1 debe calcular τ con la densidad del motor PR, no con PV = nRT.**

Trampas de la API de DWSIM anotadas: `DefineKineticReaction` exige órdenes para TODOS los compuestos de la estequiometría (también 0); E en J/mol; CSTR y PFR necesitan corriente de energía en el puerto 1 incluso en modo isotérmico; `CSTRBuilder.WithIsothermalTemperature` rompe el flash PH (T → 3·10⁷ K) — usar `Isothermal()` con la alimentación a la T deseada.

## B. Metano + aire estequiométrico (GRI-Mech 3.0, 53 especies / 325 reacciones)

### B1. Temperatura de llama adiabática: equilibrio real vs conversión 100 %

| Modelo | T adiabática | Qué supone |
|---|---|---|
| Cantera, equilibrio HP (GRI-Mech) | **2226 K** | 53 especies; disociación a CO, H₂, OH, O, H, NO… |
| DWSIM, reactor de conversión adiabático 100 % (PR) | **2341 K** | solo CH₄ + 2 O₂ → CO₂ + 2 H₂O; cp reales de PR; balances cerrados = True |
| Diferencia | **+116 K (+5.2 %)** | la disociación absorbe entalpía: el reactor de conversión sobreestima |

Composición de equilibrio (fracción mol): CO2 0.0854, H2O 0.1835, CO 0.0090, O2 0.0046, H2 0.0036, OH 0.0029, NO 0.0019, N2 0.7086.

Esto es lo que debe decir Fluxo cuando alguien use el reactor de conversión para una llama: «sin disociación, la T es un tope superior».

### B2. Reactor bien agitado (WSR) adiabático: la cinética manda cuando τ es corto

Alimentación 300 K, 1 atm, estequiométrica. El reactor arranca en equilibrio (encendido) y se lleva a estado estacionario para cada τ. Si el tiempo de residencia es menor que el tiempo químico, la llama se apaga (blow-out): la conversión cae a ~0 aunque el equilibrio diga 2225 K.

| τ (ms) | T salida (K) | X CH₄ | CO (% mol) | NO (ppm) | Estado |
|---|---|---|---|---|---|
| 0.05 | 300 | 0.000 | 0.00 | 0 | apagado |
| 0.1 | 300 | 0.000 | 0.00 | 0 | apagado |
| 0.2 | 300 | 0.000 | 0.00 | 0 | apagado |
| 0.5 | 1947 | 0.998 | 2.71 | 97 | encendido |
| 1 | 2000 | 0.999 | 2.42 | 136 | encendido |
| 3 | 2075 | 0.999 | 1.99 | 222 | encendido |
| 10 | 2139 | 1.000 | 1.57 | 360 | encendido |
| 30 | 2181 | 1.000 | 1.27 | 544 | encendido |
| 100 | 2208 | 1.000 | 1.06 | 851 | encendido |

Blow-out entre τ = 0.2 y 0.5 ms. Con τ largo el WSR tiende al equilibrio (2208 K vs 2226 K) y el NO sube con τ (cinética de Zeldovich, lenta): 97 → 851 ppm. Ningún reactor de conversión ni de equilibrio puede dar esta curva.

## C. Retardo de ignición (batch adiabático a volumen constante)

| T₀ (K) | P₀ (bar) | τ_ign (ms) por máximo de dT/dt |
|---|---|---|
| 1000 | 10 | 80.870 |
| 1200 | 10 | 4.520 |
| 1400 | 10 | 0.470 |
| 1200 | 40 | 0.968 |

El retardo cae ~un orden de magnitud por cada +200 K (Arrhenius global) y baja con la presión. Útil para dimensionar cámaras y para el HAZOP (mezcla inflamable + punto caliente = ignición en milisegundos).

## D. Qué necesita el reactor cinético de Fluxo 1.1 (derivado de A–C)

1. **Ley de velocidad**: Arrhenius con ley de potencias, r = A·exp(−E/RT)·Π C_i^n_i, concentraciones en mol/m³ desde la
   densidad del motor PR/SRK (no gas ideal). Reversible opcional con (A_r, E_r). Unidades explícitas en la UI (A en 1/s para
   primer orden, E en kJ/mol).
2. **CSTR**: estado estacionario, sistema algebraico F·(C_in − C_out) + r(C_out, T)·V = 0 por compuesto; Newton con jacobiano
   numérico; isotérmico (Q = ΔH) y adiabático (T de salida por balance de energía con ΔHf del motor, misma convención que el
   reactor de conversión actual).
3. **PFR**: ODE en el volumen dF_i/dV = ν_i·r; RK4 adaptativo (o Dormand-Prince) con 200+ puntos para el perfil; isotérmico y
   adiabático (dT/dV por balance de energía).
4. **Validación obligatoria antes de mostrar al usuario**: este caso A→B (X_PFR = 0.9807, X_CSTR = 0.7979 a kτ = 3.948) dentro de
   0.1 %; combustión 100 % vs reactor de conversión; aviso «sin disociación: T tope superior» cuando T > 1500 K.
5. **Fuera de alcance 1.1**: mecanismos detallados (GRI-Mech), catalítico heterogéneo, no isotérmico con intercambio (modo
   HeatExchange). Eso queda en AlchemAI vía DWSIM/Cantera.

Evidencia reproducible: `uv run python notebooks/03_cantera_reactores.py` · tests `tests/test_kinetic_reactors.py`.
