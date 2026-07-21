# ADR-004: PDF único sin código, concebido como presentación

**Estado:** Aceptada
**Fecha:** 2026-07-20 (validada vía grill-me, pregunta 4)
**Decisores:** Max
**Relacionados:** [[SPEC]] R1, R6 · [[PLAN]] F5

## Contexto

El profesor pidió "un PDF para exponer", ambiguo entre informe técnico y soporte de presentación. Max decidió usar el PDF como material de exposición proyectado. El notebook completo (con código y outputs) existe igualmente y se lleva abierto en Colab como respaldo.

## Decisión

Un único PDF generado con `nbconvert --to webpdf --no-input`, excluyendo además las celdas con tag `no-pdf` (`TagRemovePreprocessor`). El código no aparece en el PDF; la prosa y las figuras cargan el 100% de la narrativa.

## Opciones consideradas

| Opción | Legibilidad al proyectar | Cobertura "muéstrenme el código" | Costo |
|---|---|---|---|
| A. PDF completo con código | Baja (paredes de código en pantalla) | Total | Un comando |
| **B. PDF `--no-input` (elegida)** | **Alta** | **Vía notebook en Colab como respaldo** | **Un comando** |
| C. Ambos PDFs | Alta | Total | Dos artefactos que mantener coherentes |

## Análisis de trade-offs

B contra C: mantener dos PDFs coherentes durante las iteraciones de F5 es fricción sin beneficio claro — el escenario "quiero ver la implementación" se cubre mejor con el notebook vivo (ejecutable, con outputs) que con un PDF estático. B contra A: en exposición, cada bloque de código proyectado compite por atención contra el expositor.

## Consecuencias

- (+) Documento limpio donde figuras y fórmulas respiran.
- (−) **Endurece R1**: prohibida toda referencia al código desde la prosa; cada figura necesita epígrafe autoexplicativo (verificado en R6, lectura fría).
- (−) La demostración de implementación depende de tener Colab accesible en la defensa; mitigación: notebook pre-ejecutado, sin depender de re-correr en vivo.
- (⚠) `webpdf` requiere Chromium local; fallback documentado en Makefile (`--to html` + print del navegador).

## Acciones

1. [ ] F0: target `make pdf` con `--no-input` + `TagRemovePreprocessor` para `no-pdf`.
2. [ ] F5: prueba de lectura fría (R6) como gate del PDF final.
