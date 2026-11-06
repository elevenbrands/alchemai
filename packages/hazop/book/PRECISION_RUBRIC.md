# Rúbrica de precisión del HAZOP preliminar (bloque 25)

Escrita el 2026-11-05 **antes** de correr las reglas sobre el conjunto apartado (`heldout.yaml`). No se cambia después.

Cada fila que genera AlchemAI (nodo · parámetro · desviación, con sus causas, consecuencias y salvaguardas) recibe
una sola etiqueta. La califica un revisor que ve la descripción del proceso y las filas, pero no las reglas ni la
referencia publicada (para no premiar el parecido de redacción).

| Etiqueta | Cuándo |
|---|---|
| **coincide** | El calificador automático la empareja con una desviación del HAZOP publicado (`score.find`). |
| **plausible** | No está en el HAZOP publicado, pero un ingeniero de procesos la aceptaría en la sesión: la causa puede ocurrir en ese equipo tal como está descrito y la consecuencia es coherente con T, P y composición. |
| **ruido** | Correcta pero sin valor: repite otra fila del mismo nodo con la misma causa, o es tan genérica que no dice nada del proceso. |
| **espuria** | Errónea: la causa no puede ocurrir en ese equipo, la consecuencia contradice los datos (ej. inflamable cuando es agua, congelamiento a 150 °C) o la salvaguarda propuesta empeora el escenario. |

Métricas reportadas:
- **Precisión estricta** = coincide / total. Es una cota inferior: los HAZOP publicados no son exhaustivos.
- **Precisión útil** = (coincide + plausible) / total. Es la que importa para el usuario: cuántas filas aceptaría.
- **Tasa de espurias** = espuria / total. Meta del plan: < 5 %.

Recall se mide igual que en el bloque 17 (`alcance: datos` y todos), con la referencia escrita antes de correr.
Las reglas **no** se tocan entre la referencia y la primera corrida; si después se corrigen, la corrida a ciegas queda
como cifra oficial y la corregida se reporta aparte.
