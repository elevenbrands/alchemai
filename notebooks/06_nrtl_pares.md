# Los 16 pares NRTL de FluxoEngine vs DWSIM 10 — Txy a 1 atm (semana 5b)

Objetivo del bloque 12: validar los 15 pares NRTL que quedaron transcritos sin prueba en el bloque 11. Referencia:
DWSIM 10.2.12, paquete NRTL, punto de burbuja (VF = 0) en 11 composiciones x = 0.05…0.95 por par
(`06_nrtl_pares_dwsim.py` → `NRTLParesDWSIM.swift` generado por `06_nrtl_pares_gen.py`; test `NRTLParesTests`).

Dos niveles por par: **(a)** Fluxo con la Psat de DWSIM (ChemSep) y vapor ideal, que debe reproducir a DWSIM si el
algoritmo y la transcripción son correctos; **(b)** Fluxo con sus datos (Psat de Perry's, vapor PR), donde la diferencia
es la de las fuentes de datos. La columna «Python» es una réplica independiente de γ-φ con las constantes de DWSIM
(bisección a 1e-10): si coincide con Fluxo, el residuo frente a DWSIM es de DWSIM.

| Par (i\|j) | A12 / A21 (cal/mol) | α | ΔT máx (a) K | Δy máx (a) | ΔT máx Python K | ΔT máx (b) K | Δy máx (b) |
|---|---|---|---|---|---|---|---|
| etanol|agua | -57.9601 / 1241.74 | 0.2937 | 0.014 | 0.0002 | 0.014 | 0.331 | 0.0021 |
| metanol|agua | -189.047 / 792.802 | 0.2999 | 0.027 | 0.0005 | 0.027 | 0.231 | 0.0062 |
| metanol|etanol | -327.999 / 376.267 | 0.3057 | 0.013 | 0.0001 | 0.013 | 0.310 | 0.0053 |
| etanol|benceno | 516.141 / 1065.91 | 0.4774 | 0.026 | 0.0001 | 0.026 | 0.436 | 0.0013 |
| etanol|tolueno | 713.565 / 1147.86 | 0.5292 | 0.046 | 0.0005 | 0.046 | 0.547 | 0.0055 |
| etanol|n-heptano | 1137.27 / 1453.59 | 0.4786 | 0.041 | 0.0007 | 0.041 | 0.575 | 0.0038 |
| etanol|n-octano | 1206.81 / 1385.37 | 0.4717 | 0.059 | 0.0008 | 0.059 | 0.538 | 0.0080 |
| n-pentano|etanol | 1183.38 / 412.755 | 0.2886 | 0.102 | 0.0016 | 0.102 | 0.578 | 0.0233 |
| n-hexano|etanol | 1218.11 / 575.298 | 0.2882 | 0.064 | 0.0037 | 0.064 | 0.671 | 0.0043 |
| metanol|benceno | 721.614 / 1158.51 | 0.4694 | 0.025 | 0.0004 | 0.025 | 0.324 | 0.0037 |
| metanol|tolueno | 939.727 / 1090.93 | 0.4643 | 0.050 | 0.0005 | 0.050 | 0.286 | 0.0108 |
| metanol|n-hexano | 1619.38 / 1622.29 | 0.4365 | 0.058 | 0.0017 | 0.058 | 0.601 | 0.0079 |
| metanol|n-heptano | 1313.55 / 1143.91 | 0.4163 | 0.047 | 0.0016 | 0.047 | 0.390 | 0.0081 |
| metanol|n-octano | 1681.69 / 1511.44 | 0.4381 | 0.037 | 0.0006 | 0.037 | 0.245 | 0.0078 |
| n-butano|metanol | 1440.15 / 1053.77 | 0.4647 | 0.117 | 0.0006 | 0.117 | 1.083 | 0.0096 |
| benceno|tolueno | 60.198 / -51.0865 | 0.3019 | 0.059 | 0.0005 | 0.059 | 0.287 | 0.0083 |

Peor caso (a): 0.117 K; peor caso de la réplica Python: 0.117 K; peor caso (b): 1.083 K (n-butano|metanol,
sistema parcialmente miscible donde la Psat del butano manda).

## Lectura

- (a) y Python coinciden a 0.001 K en los 16 pares: Fluxo calcula exactamente γ·Psat/P con los parámetros transcritos.
  El residuo frente a DWSIM (≤ 0.12 K, siempre con DWSIM por debajo, mayor donde dT/dP es grande) es la tolerancia
  del flash de DWSIM (≈ 2·10⁻³ en presión); Fluxo converge a 1e-10. Un A12/A21 invertido o una cifra mal copiada
  darían varios kelvin: los 16 pares están bien transcritos.
- (b) queda entre 0.23 y 1.08 K: es la diferencia entre la Psat de Perry's y la de ChemSep (p. ej. etanol 351.47 vs
  351.8 K) más el vapor PR; no es un defecto del modelo. Para reproducir a DWSIM se puede inyectar su Psat (como hace
  el test); para estar bien, la de Perry's es la correcta en los puros que se comprobaron.
- Varios pares (metanol con alcanos C4–C8) son parcialmente miscibles a 1 atm: los dos motores calculan VLE sin
  separación de líquidos, así que la comparación valida los parámetros, no la física del azeótropo heterogéneo.
  Fluxo no tiene VLLE (anotado en MOCKS).
