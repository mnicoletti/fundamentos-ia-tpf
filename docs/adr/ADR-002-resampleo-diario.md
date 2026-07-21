# ADR-002: Resampleo del dataset 5-min a frecuencia diaria

**Estado:** Aceptada
**Fecha:** 2026-07-19 (formalizada 2026-07-20)
**Decisores:** Max
**Relacionados:** [[SPEC]] · [[ADR-005-activo-canonico-reliance]]

## Contexto

La consigna usa el dataset Kaggle "Nifty 100 — 5 min". A esa frecuencia, el ruido microestructural (spread, discretización de ticks, estacionalidad intradía) domina cualquier señal direccional, y los indicadores técnicos canónicos (RSI-14, SMA-20, Bollinger-20) están definidos y calibrados sobre barras diarias. Los hiperparámetros pedagógicos del experimento (ventana de 30 períodos, horizonte de 5) solo son interpretables si el período es un día.

## Decisión

Cargar el CSV 5-min y resamplear a velas diarias con agregaciones OHLCV correctas (`first/max/min/last/sum`). El resto del pipeline opera en diario. `yfinance` queda como fallback reproducible si el CSV no está disponible.

## Opciones consideradas

| Opción | Ruido | Interpretabilidad | Fidelidad a la consigna | Cómputo |
|---|---|---|---|---|
| A. Operar en 5-min nativo | Extremo | Baja (k=5 barras = 25 min) | Máxima | Alto (~10⁵ filas/activo) |
| **B. Resamplear a diario (elegida)** | **Moderado** | **Alta** | **Alta (mismo dataset)** | **Bajo** |
| C. Ignorar el dataset, usar solo yfinance | Moderado | Alta | Nula (inconsistencia detectable) | Bajo |

## Consecuencias

- (+) Hiperparámetros defendibles en lenguaje natural ("5 días vista").
- (+) El dataset declarado en el TP es el que efectivamente alimenta el modelo.
- (−) Se descarta información intradía; se declara explícitamente como decisión (no como limitación oculta) y el análisis intradía queda como trabajo futuro.
- (⚠) El resampleo no introduce leakage (cada vela diaria usa solo información de su día), pero debe afirmarse en el texto porque es pregunta esperable.

## Acciones

1. [x] Loader dual implementado y verificado con smoke test (2026-07-19).
2. [ ] F1: justificación metodológica del resampleo redactada en la sección de datos.
