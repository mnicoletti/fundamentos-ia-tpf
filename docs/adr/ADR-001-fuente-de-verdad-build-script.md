# ADR-001: `build_nb.py` como fuente de verdad del notebook

**Estado:** Aceptada
**Fecha:** 2026-07-20
**Decisores:** Max (validado vía grill-me, pregunta 5)
**Relacionados:** [[SPEC]] · [[PLAN]]

## Contexto

El `.ipynb` es JSON con outputs embebidos: los diffs en Git son ilegibles, la edición programática ad-hoc (scripts `nbformat` descartables, como en las dos primeras iteraciones) no deja rastro versionable, y los agentes de código tienden a corromper notebooks al editarlos como texto. El proyecto requiere una reescritura masiva de prosa (R1/R4 tocan todas las celdas markdown) que se hará mayormente con Claude Code.

## Decisión

Un script Python versionable `build_nb.py` declara cada celda como literal y genera el `.ipynb` con `nbformat` (`make build`). El notebook generado es un *build artifact* (directorio `dist/`, gitignored salvo la versión congelada de entrega). Nadie —humano ni agente— edita el `.ipynb` a mano.

## Opciones consideradas

| Opción | Diffs limpios | Riesgo de corrupción | Costo operativo | Familiaridad |
|---|---|---|---|---|
| A. Editar el `.ipynb` directo | No | Alto (agentes sobre JSON) | Nulo | Alta |
| **B. `build_nb.py` (elegida)** | **Sí** | **Nulo** | **Paso de build** | **Alta (patrón ya usado)** |
| C. Par jupytext (`.py` percent) | Sí | Bajo | Sincronización bidireccional | Media |

## Análisis de trade-offs

B contra C: jupytext es elegante para proyectos donde el notebook se edita interactivamente y debe re-sincronizarse al `.py`; acá el flujo es unidireccional (fuente → artefacto → Colab solo ejecuta) y el proyecto vive dos semanas — la dependencia extra no paga. B contra A: el paso de build es el precio de que `str_replace` opere sobre texto plano con precisión y de que cada cambio de prosa sea un diff revisable en Gitea.

## Consecuencias

- (+) R1 (reescritura de voz) y `test_voz` operan sobre strings Python triviales de analizar.
- (+) Historia de cambios de prosa revisable commit a commit.
- (−) Toda edición requiere `make build` antes de ver el resultado; olvido típico mitigado porque `make verify` depende de `build`.
- (⚠) Los outputs de la corrida canónica (F4) viven solo en la copia congelada de `dist/`; se commitea explícitamente esa única versión para la entrega.

## Acciones

1. [ ] F0: migrar el `.ipynb` actual a `build_nb.py` (una función por sección R8).
2. [ ] `Makefile`: `build` → `verify` encadenados.
