# ADR-003: Clasificación de 3 clases con zona muerta (vs. detector binario)

**Estado:** Aceptada
**Fecha:** 2026-07-20 (validada vía grill-me previo)
**Decisores:** Max
**Relacionados:** [[SPEC]] · guía de defensa §4

## Contexto

La consigna admite "detectores binarios" o "de múltiples características (N categorías)". El profesor pide caracterizar con curva ROC, curva mAP y matrices de confusión. Con 3 clases, la ROC exige el esquema one-vs-rest; con 2, es una curva única más simple de exponer. La versión de 3 clases ya está implementada y verificada.

## Decisión

Mantener 3 clases — SUBE / LATERAL / BAJA con zona muerta ±θ — y caracterizar con ROC y Precision-Recall one-vs-rest por clase (AUC macro y mAP como promedio macro de AP).

## Opciones consideradas

| Opción | Riqueza analítica | Simplicidad expositiva | Costo de implementación | Solidez del etiquetado |
|---|---|---|---|---|
| A. Binario (¿sube >θ? sí/no) | Baja (baja fuerte ≡ lateral) | Alta | Rehacer etiquetado, métricas y análisis | Media |
| **B. 3 clases (elegida)** | **Alta** | **Media (one-vs-rest requiere explicación)** | **Nulo (ya verificado)** | **Alta (la zona muerta filtra ruido ±0.1%)** |

## Análisis de trade-offs

El costo real de B es explicar one-vs-rest — un párrafo, ya escrito en §10.1. A cambio: la zona muerta es uno de los mejores argumentos de diseño en la defensa (sin ella, ruido de ±0.1% contaría como señal), las matrices de confusión 3×3 dan material interpretativo (¿confunde LATERAL con SUBE? ¿confunde SUBE con BAJA?) que una matriz 2×2 no ofrece, y elegir A implicaría descartar trabajo verificado a diez días del deadline.

## Consecuencias

- (+) R5 (análisis por clase) tiene sustancia real.
- (+) Cumple la variante "N categorías" de la consigna de forma directa.
- (−) El PDF debe incluir el mini-marco de one-vs-rest (absorbido por R4).

## Acciones

1. [x] ROC/PR one-vs-rest implementadas y verificadas (§10.1, 2026-07-19).
2. [ ] F3: mini-marco "one-vs-rest y por qué mAP = promedio de AP" (R4).
