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
- (⚠) Gate en F0: validar el CSV real (rango, huecos, volumen) *antes* de escribir prosa dependiente del activo; si falla, este ADR se actualiza a HDFCBANK con la evidencia del fallo registrada.
- (−) Los resultados de un solo activo no generalizan; se declara en limitaciones (multi-ticker = trabajo futuro).

## Acciones

1. [ ] F0: descargar y validar `RELIANCE_*.csv` (script de validación en `verify_pipeline.py`).
2. [ ] F1: párrafo de pre-registro en la sección de datos.
