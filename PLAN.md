# PLAN — TP Fundamentos de la IA

**Ventana:** 2026-07-20 → 2026-07-30 (10 días, con período de finales en paralelo)
**Artefactos:** [[SPEC]] · ADRs en `docs/adr/`
**Principio rector:** cada fase deja el repo en estado entregable-degradado. Si el tiempo se acaba en cualquier punto después de F1, hay algo presentable.

---

## F0 — Scaffold y migración (20/07, ~2h)

Objetivo: repo funcionando con la fuente de verdad correcta ([[ADR-001-fuente-de-verdad-build-script]]).

1. Crear repo en Gitea: `tp-fundamentos-ia` con la estructura declarada en `CLAUDE.md`.
2. Migrar el `.ipynb` actual (41 celdas, ya corregido con ROC/PR y loader dual) a `build_nb.py`: una función por sección, cada celda como string literal.
3. `Makefile` con targets `build`, `verify`, `pdf`, `clean`.
4. Portar el smoke test existente a `verify_pipeline.py` (CSV sintético GBM + ejecución de celdas con MLP mockeado + asserts de alineación).
5. Descargar y validar `RELIANCE_*.csv` del dataset Kaggle: rango temporal, huecos, volumen. Si falla → activar suplente HDFCBANK y actualizar [[ADR-005-activo-canonico-reliance]].

**Gate de salida:** `make build && make verify` en verde; notebook generado byte-idéntico en contenido al actual.

## F1 — Reescritura de voz y estructura (21–22/07)

Objetivo: R1 + R8 completos. Es la fase de mayor volumen de texto y la de menor riesgo técnico — ideal para delegar a Claude Code con revisión tuya.

1. Reordenar secciones según R8 (la actual §2 "Configuración" → celda técnica `no-pdf`; la actual §3 "Parámetros" → "3. Diseño experimental" con notación θ, k, w y justificaciones).
2. Reescribir toda celda markdown en registro académico (prohibiciones de R1).
3. Implementar `test_voz` en `verify_pipeline.py` (grep de términos prohibidos sobre las fuentes markdown).
4. Epígrafes interpretativos en todas las figuras ya existentes.

**Gate de salida:** `test_voz` en verde; lectura tuya de corrido sin encontrar voz de asistente.

## F2 — EDA, figuras didácticas y sensibilidad (22–24/07)

Objetivo: R2 + R3 + R7. El grueso del código nuevo.

1. Sección EDA completa (4 evidencias con epígrafes — R2).
2. Figura de etiquetado ±θ sobre el precio (R3.1).
3. Timeline del split purgado (R3.2).
4. Anotaciones sobre curvas de pérdida (R3.3).
5. Figura de sensibilidad θ → distribución de clases + párrafo de justificación de θ=2% (R7).
6. Extender `verify_pipeline.py` para cubrir las celdas nuevas.

**Gate de salida:** `make verify` en verde con las celdas nuevas incluidas.

## F3 — Mini-marcos teóricos (24–25/07)

Objetivo: R4. **Fase de estudio, no solo de escritura** — el objetivo secundario declarado es que entiendas la materia.

1. Redactar los 8 bloques Concepto (intuición → fórmula → término por término → anclaje → cita).
2. Loop de validación por bloque: leés el bloque → explicás la fórmula en voz alta sin mirarla → si no podés, el bloque se reescribe (el problema es del texto, no tuyo).
3. Checklist R4 agregada a la guía de defensa.

**Gate de salida:** checklist R4 completa: 8/8 conceptos explicables en voz alta.

## F4 — Corrida canónica y análisis real (26/07)

Objetivo: resultados reales congelados + R5.

1. `make build` → subir a Colab → corrida completa con `RELIANCE`.
2. Verificar asserts de alineación y coherencia de todas las figuras con datos reales.
3. Escribir R5 (análisis por clase) **sobre los números reales de esta corrida** — es la única sección que no puede escribirse antes.
4. Sanity check final con paseo aleatorio documentado en el texto (una línea + referencia).
5. Congelar: el `.ipynb` con outputs de esta corrida es el que se entrega y del que sale el PDF.

**Gate de salida:** notebook ejecutado de punta a punta en Colab, cero errores, R5 escrito.

## F5 — PDF, guía de defensa y ensayo (27/07)

1. `make pdf` (`--no-input`, tags `no-pdf` excluidos). Revisión tipográfica: cortes de página, tamaño de figuras, legibilidad de fórmulas.
2. **Prueba de lectura fría** (R6): lectura completa del PDF sin abrir el notebook. Todo hueco encontrado → fix en `build_nb.py` → rebuild → re-export.
3. Actualizar guía de defensa: cada respuesta anticipada acompañada de su razonamiento (por qué esa respuesta es cierta), no solo el texto a recitar.
4. Ensayo de exposición de 15 min contra el guion, con el PDF proyectado.

**Gate de salida:** PDF final + guía actualizada + un ensayo completo cronometrado.

## Buffer — 28–29/07

Sin tareas asignadas por diseño. Absorbe: finales que se corran, hallazgos de la lectura fría, comentarios tardíos del profesor (dijo que responde mensajes el finde). Si el buffer no se usa: segundo ensayo de exposición.

## Entrega — 30/07

Notebook (`.ipynb` con outputs) + PDF, por el canal que indique la cátedra.

---

## Riesgos y mitigaciones

| Riesgo | Prob. | Impacto | Mitigación |
|---|---|---|---|
| CSV de RELIANCE corto o con huecos | Media | Medio | Suplente HDFCBANK validado en F0, no el 26 |
| Finales comprimen F2–F3 | Alta | Alto | F1 delegable a Claude Code casi por completo; R7 y R3 son independientes entre sí → recortables individualmente sin romper el resto |
| `webpdf` falla en tu entorno local | Baja | Bajo | Fallback documentado en Makefile: `--to html` + print-to-PDF del navegador |
| Corrida real contradice prosa escrita en F1–F3 | Media | Medio | Toda prosa pre-F4 evita números concretos de resultados; solo R5 los usa |
| El profesor formaliza una consigna distinta | Baja | Alto | La estructura R8 cubre el superset de lo pedido informalmente (ROC, mAP, matrices, PDF); desvíos se absorben en buffer |
