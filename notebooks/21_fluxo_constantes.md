# 21 · Constantes de los 24 componentes de Fluxo contra fuentes de referencia (bloque 30, 2026-11-06)

**Pregunta.** Las constantes de Fluxo (Poling et al. 5.ª ed., Cp de Reid et al.) se transcribieron a mano. ¿Coinciden
con datos críticamente evaluados? Es el pendiente 🟠 más viejo de `apps/fluxo/MOCKS.md`.

**Método.** `notebooks/21_fluxo_constantes.py` lee la tabla directamente del Swift de FluxoEngine y compara cada
constante con una fuente de referencia, con tolerancias fijadas antes de correr: MW ±0.02, Tc ±0.5 %, Pc ±1 %,
ω ±0.01, Tb ±0.5 K, ΔHf° ±1 kJ/mol, Cp° ±2 % a 300, 500 y 800 K.

## Resultados

| Corrida | Orden de fuentes | Dentro de tolerancia |
|---|---|---|
| 1 | NIST WebBook primero; Cp contra el ajuste HEOS_FIT | 150/168 |
| 2 (oficial) | Tc/Pc: IUPAC → ec. de referencia (CoolProp) → WebBook · Tb, ω: ec. de referencia → CRC/PSRK · ΔHf°: ATcT → WebBook · Cp°: TRC → JANAF → CoolProp | **166/167** |

Cada una de las 18 alertas de la corrida 1 se revisó contra dos o tres fuentes más:
- **Errores de la fuente, no de Fluxo:** las entradas de WebBook dentro de `chemicals` para Pc del argón (30.65 bar; el
  real es 48.6), Tc del CO (134.45 K; real 132.86), Tb del metano (111.0 K; real 111.67), Tb del isobutano (262 K;
  real 261.40) y ΔHf° del metanol (−205 kJ/mol; ATcT −200.7). Fluxo coincide con las otras fuentes.
- **Cp:** el ajuste HEOS_FIT no vale a 800 K (da 69 107 J/mol/K para etileno). Contra TRC, JANAF y CoolProp los
  polinomios de Fluxo quedan dentro de ±1 %.
- **Una corrección real:** ω del H₂S, 0.090 (Poling) vs 0.1005 de la ecuación de referencia de Lemmon y Span (2006).
  Fluxo pasa a 0.100. Ninguna prueba del motor usa H₂S; las 57 pruebas de FluxoEngine y las 41 de la app siguen verdes.
- **Sin referencia:** la Tb del CO₂ (194.70 K) es el punto de sublimación a 1 atm; el CO₂ no tiene ebullición normal.
- **Hidrógeno:** Fluxo usa los valores de hidrógeno de equilibrio de Poling (Tc 32.98 K); el normal es 33.145 K
  (−0.5 %, en el límite de la tolerancia). Sin efecto práctico a temperatura de proceso.

**Límites.** La comparación pasa por la librería `chemicals` (MIT), que agrega las fuentes; no se consultó NIST
WebBook en línea constante por constante. Los k_ij y la densidad de Rackett siguen como pendientes aparte en MOCKS.

Archivos: `notebooks/21_fluxo_constantes.json` (corrida oficial), `notebooks/21_fluxo_constantes_corrida1.json`.
