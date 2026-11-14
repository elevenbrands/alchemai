# Post 2 — El error que encontramos en DWSIM (borrador para LinkedIn, español)

> Estado: BORRADOR 2026-11-14. No publicado. Lo publica el owner si lo aprueba.
> Hechos verificados el 2026-11-14: issue https://github.com/DanWBR/dwsim10/issues/137 (abierto el 13 de noviembre por
> la noche, cerrado como completado el 14 por la mañana); commit `901b173` de DWSIM leído en su código fuente (los seis
> modelos usan ya la derivada de ln γ). No verificado: la corrección no se ha vuelto a medir porque sale en la 10.2.17.

---

Esta semana mi agente de IA encontró un error en un simulador de procesos que se usa en todo el mundo. Lo reporté de noche y a la mañana siguiente ya estaba corregido.

Así pasó.

Estoy construyendo **AlchemAI**, un agente para ingeniería de procesos con una regla: ningún número sale sin prueba. Cada resultado se compara contra otro motor o contra la literatura.

Al comparar una columna de etanol-agua contra **DWSIM**, la carga del rehervidor no cuadraba: unos 56 kW de diferencia en una carga de 1 MW. Con metanol-agua, en cambio, coincidía a 0.6 %.

Esa pista llevó al origen. En las mezclas muy no ideales, el calor de mezcla que DWSIM sumaba a la entalpía del líquido salía entre 1.4 y 4.6 veces más grande de lo que da la termodinámica. Ejemplo: etanol-agua a 25 °C con 10 % mol de etanol daba 1 125 J/mol; la relación de Gibbs-Helmholtz da 363.

La causa era una sola línea: el código derivaba γ respecto a la temperatura, y la ecuación pide derivar **ln γ**. El mismo patrón estaba en seis modelos: NRTL, UNIQUAC, UNIFAC, UNIFAC modificado, NIST-MFAC y Wilson.

Lo reporté con la tabla, la línea exacta y un script de 20 líneas para reproducirlo. El equipo de DWSIM lo confirmó y corrigió los seis modelos en menos de diez horas. Sale en su próxima versión.

Tres cosas que quiero dejar claras:

1. **DWSIM es un gran proyecto.** Es gratuito, abierto, y su equipo respondió en horas. El error solo afectaba dos modos de entalpía que no son los de fábrica.
2. **Esto no lo encontró «la IA» adivinando.** Lo encontró la disciplina de comprobar: dos motores, los mismos parámetros, y no aceptar una diferencia sin explicación.
3. **Por eso existe AlchemAI.** Si un simulador maduro puede traer un error así, un modelo de lenguaje que responde «con seguridad» sin comprobar no sirve para ingeniería.

El reporte completo y el código son públicos:
- Issue: https://github.com/DanWBR/dwsim10/issues/137
- AlchemAI (MIT): https://github.com/elevenbrands/alchemai

Si trabajas en una planta y alguna vez un número del simulador no te cuadró con la realidad, me interesa tu historia.

#IngenieríaQuímica #IngenieríaDeProcesos #DWSIM #Termodinámica #IA #CódigoAbierto

---

Notas para el owner:
- Tono: agradecido con DWSIM, no de «los caché». Si se cambia el texto, conservar el punto 1.
- Las cifras salen del issue publicado (tabla de 12 estados y el ejemplo de la columna).
- Posible imagen: la tabla del issue (DWSIM contra Gibbs-Helmholtz) o la línea de código antes y después.
- La respuesta pública en el issue la firma «the AI assistant working with Daniel»; aquí se dice «el equipo de DWSIM».
- Antes de publicar: confirmar si la versión 10.2.17 ya salió y, si sí, decirlo.
