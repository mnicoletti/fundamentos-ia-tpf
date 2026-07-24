# ADR-005: RELIANCE como activo canónico pre-registrado

**Estado:** Aceptada
**Fecha:** 2026-07-20 (validada vía grill-me, pregunta 3)
**Decisores:** Max
**Relacionados:** [[SPEC]] D3 · [[ADR-002-resampleo-diario]] · [[PLAN]] F0

## Contexto

El loader acepta cualquier CSV del dataset Nifty 100, pero las figuras del PDF, los números del análisis (R5) y la demo deben salir de un único activo. Elegirlo *después* de ver resultados sería sesgo de selección (cherry-picking) — exactamente el tipo de deshonestidad metodológica que el TP dice combatir.

## Decisión

RELIANCE como activo canónico, decidido **antes** de correr ningún experimento sobre datos reales (pre-registro). Suplente documentado: HDFCBANK, activable solo por criterio de calidad de datos (rango corto o huecos), nunca por resultados.

## Opciones consideradas

| Opción | Justificabilidad | Riesgo de sesgo | Calidad de datos esperada |
|---|---|---|---|
| **A. RELIANCE (elegida)** | **Una línea: mayor liquidez del índice ⇒ precio más informativo** | **Nulo (pre-registrado)** | **Alta** |
| B. HDFCBANK | Similar (large-cap bancario) | Nulo | Alta |
| C. Correr varios y elegir | — | Alto (cherry-picking) | — |

## Consecuencias

- (+) El pre-registro es en sí un punto metodológico defendible y se declara en el texto.
- (+) Justificación autocontenida, sin requerir contexto sectorial indio.
- (−) Los resultados de un solo activo no generalizan; se declara en limitaciones (multi-ticker = trabajo futuro).

## Validación del CSV real (F0, 2026-07-23)

`data/RELIANCE_5minute.csv` (206.829 filas, 5 min, `date,open,high,low,close,volume`) pasó el gate sin activar el suplente:

| Chequeo | Resultado |
|---|---|
| Rango temporal | 2015-02-02 → 2026-04-09 (~11 años) |
| Timestamps duplicados / no monótonos | 0 |
| Nulos | 0 |
| Precios ≤ 0 / volumen < 0 | 0 |
| Volumen == 0 | 3 filas (0.001%) |
| Barras fuera de sesión 09:15–15:30 | 134 (0.06%) — *muhurat trading* (sesión especial nocturna de Diwali, real, no error) |
| Barras por día (mediana) | 75 (= 6h15 / 5 min, consistente con sesión NSE completa) |
| Días hábiles sin datos | 160 / 2919 (5.5%) — consistente con el calendario de feriados NSE, no huecos anómalos |
| Filas diarias tras resample + dropna | 2.770 |

Conclusión: sin huecos anómalos, sin datos corruptos, volumen y rango muy por encima del mínimo necesario para `VENTANA=30`, `HORIZONTE=5`, `TEST_FRAC=0.20`. **RELIANCE queda confirmado como activo canónico; no se activa HDFCBANK.**

## Acciones

1. [x] F0: descargar y validar `RELIANCE_*.csv` (2026-07-23, evidencia arriba).
2. [ ] F1: párrafo de pre-registro en la sección de datos.
