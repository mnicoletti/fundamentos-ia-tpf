# SPEC — TP Fundamentos de la IA: Clasificación de dirección de precios (LogReg vs. MLP)

**Tipo:** SKS (Spec de alcance y contenido)
**Estado:** Confirmado — todas las decisiones validadas vía grill-me el 2026-07-20
**Deadline duro:** 2026-07-30 (entrega + exposición)
**Artefactos relacionados:** [[PLAN]] · [[ADR-001-fuente-de-verdad-build-script]] · [[ADR-002-resampleo-diario]] · [[ADR-003-clasificacion-tres-clases]] · [[ADR-004-pdf-sin-codigo]] · [[ADR-005-activo-canonico-reliance]]

---

## 1. Problema

El notebook existente (`TP_Fundamentos_IA_Financiero.ipynb`, 41 celdas) es **metodológicamente correcto pero no es un entregable académico**:

1. **Voz de narrador incorrecta**: celdas markdown escritas como instrucciones de asistente al operador ("instalamos las dependencias", "cambiar el TICKER permite experimentar") en lugar de prosa académica dirigida al evaluador.
2. **Extensión corta, sin arco pedagógico**: falta EDA, faltan figuras demostrativas, el análisis de resultados no baja a nivel de clase.
3. **Teoría delegada al libro**: los conceptos de Goodfellow se citan pero no se explican; el PDF exportado no se sostiene solo.

Restricción adicional del autor: su comprensión de la materia es incompleta — **el notebook debe enseñarle a él primero**. Toda explicación teórica se escribe para un lector que parte de cero, no para demostrar erudición.

## 2. Decisiones fijadas (no re-litigar)

| # | Decisión | Valor | ADR |
|---|---|---|---|
| D1 | Profundidad matemática | Fórmulas clave con notación, sin derivaciones completas; cada fórmula explicada término por término | — (requerimiento R4) |
| D2 | Experimentos nuevos | Solo sensibilidad θ → distribución de clases (re-etiquetado, sin re-entrenamiento) | — (requerimiento R7) |
| D3 | Activo canónico | RELIANCE (suplente documentado: HDFCBANK) | [[ADR-005-activo-canonico-reliance]] |
| D4 | PDF entregable | Único, `nbconvert --no-input`, concebido como presentación | [[ADR-004-pdf-sin-codigo]] |
| D5 | Fuente de verdad | `build_nb.py` versionable; el `.ipynb` es build artifact | [[ADR-001-fuente-de-verdad-build-script]] |
| D6 | Frecuencia temporal | Dataset Kaggle Nifty 5-min resampleado a diario | [[ADR-002-resampleo-diario]] |
| D7 | Esquema de clasificación | 3 clases (SUBE/LATERAL/BAJA) con zona muerta ±θ; curvas one-vs-rest | [[ADR-003-clasificacion-tres-clases]] |
| D8 | Hiperparámetros del experimento | θ=2%, k=5 días, ventana w=30 días, split temporal 80/20 con purga=k | Heredado de la versión verificada |

## 3. Requerimientos

### R1 — Reescritura de voz (todas las celdas markdown)

Toda celda markdown en registro académico impersonal o primera persona plural ("se define", "planteamos"). Prohibiciones verificables:

- Cero referencias operativas: instalación, Colab, "cambiar X permite", "ejecutar la celda".
- Cero referencias al código desde la prosa ("como se ve en la celda anterior", "el código de arriba") — el PDF sale sin código (D4), la prosa debe sostenerse sola.
- Las notas operativas sobreviven únicamente en: (a) una celda técnica al inicio marcada con tag `no-pdf`, (b) el `CLAUDE.md` / `Makefile` del repo.

**Aceptación:** `grep` sobre las fuentes markdown en `build_nb.py` no encuentra: "instal", "Colab", "celda", "ejecut", "cambiar el", "permite experimentar" (lista exacta en `verify_pipeline.py::test_voz`). Toda celda operativa lleva tag `no-pdf`.

### R2 — Sección EDA nueva (motivación empírica de la metodología)

Sección "Análisis exploratorio" entre la carga de datos y el feature engineering, con cuatro evidencias, cada una con figura + epígrafe interpretativo:

1. Serie de precio de cierre: no estacionariedad visible (media dependiente del tiempo).
2. Serie e histograma de retornos logarítmicos vs. densidad normal: colas pesadas.
3. Autocorrelación de retornos (≈0) vs. autocorrelación de |retornos| (>0): la dirección es casi impredecible, la volatilidad no — esto justifica por qué el problema es difícil.
4. Conclusión textual explícita: la dependencia temporal invalida el supuesto i.i.d. ⇒ split temporal obligatorio (puente directo a la sección de metodología).

**Aceptación:** las 4 figuras existen, cada una con epígrafe de ≥2 oraciones que responde "qué veo y qué concluyo"; la sección cierra enlazando explícitamente con la decisión de split temporal.

### R3 — Figuras didácticas nuevas

1. **Etiquetado sobre el precio**: tramo de la serie con bandas ±θ y puntos coloreados por clase resultante (verde/gris/rojo).
2. **Timeline del split purgado**: barra horizontal train | purga | test sobre el eje temporal real, con la purga visible y anotada ("estas k muestras se descartan porque sus etiquetas miran días del test").
3. **Curvas de pérdida anotadas**: marcar la época del early stopping y la zona donde val_loss diverge de train_loss.

**Aceptación:** las 3 figuras existen con anotaciones (`annotate`/`axvline`/`axvspan`), no solo líneas peladas; epígrafes interpretativos presentes.

### R4 — Teoría autocontenida (mini-marcos conceptuales)

Bloques `> **Concepto — [nombre]**` embebidos donde el concepto se usa. Estructura obligatoria de cada bloque:

1. **Intuición** en lenguaje llano (2-4 oraciones, sin jerga previa).
2. **Fórmula** en LaTeX con notación consistente en todo el documento.
3. **Término por término**: qué es cada símbolo y por qué está en la fórmula.
4. **Anclaje**: qué figura/resultado del propio experimento lo ilustra.
5. Referencia bibliográfica al capítulo de Goodfellow (como cita, no como sustituto).

Conceptos mínimos: softmax como distribución de probabilidad · cross-entropy = −log-verosimilitud (por qué minimizar una equivale a maximizar la otra) · capacidad / under-overfitting · aproximación universal (y por qué "representar" ≠ "aprender") · backpropagation como regla de la cadena (esquemático) · L2 / dropout / early stopping como regularización · Adam (idea de tasas adaptativas, sin derivar momentos).

**Aceptación:** los 8 conceptos tienen su bloque con las 5 partes; **criterio humano**: el autor puede explicar cada fórmula en voz alta sin leerla (checklist en la guía de defensa).

### R5 — Análisis de resultados por clase

Además de la tabla agregada: lectura escrita de (a) las matrices de confusión — qué confunde cada modelo con qué y qué significa financieramente; (b) las curvas ROC/PR por clase — qué clase es más separable y por qué es esperable; (c) comparación LogReg vs. MLP en términos de "¿hay estructura no lineal aprovechable?".

**Aceptación:** ≥3 párrafos interpretativos escritos sobre los resultados **reales** de la corrida canónica con RELIANCE (no placeholders); cada afirmación referencia una figura o número concreto.

### R6 — PDF autosuficiente

El PDF (`nbconvert --to webpdf --no-input`, celdas `no-pdf` excluidas vía `TagRemovePreprocessor`) se lee de punta a punta como presentación/informe sin el notebook, sin el libro y sin la guía de defensa.

**Aceptación:** prueba de lectura fría — un lector sin contexto (o el autor tras 24h) lo lee completo y no encuentra: referencias a código, huecos conceptuales que exijan el libro, figuras sin epígrafe.

### R7 — Sensibilidad θ → distribución de clases

Figura: distribución de las 3 clases para θ ∈ {0.5%, 1%, 2%, 3%, 5%} (solo re-etiquetado). Texto que justifica θ=2% como el punto donde las tres clases retienen masa suficiente para aprender.

**Aceptación:** figura + párrafo de justificación; tiempo de cómputo <10s (si requiere re-entrenar, está mal implementado).

### R8 — Estructura académica del documento

Orden final del notebook/PDF: Portada y resumen → 1. Introducción y problema (T/P/E) → 2. Datos y EDA (R2) → 3. Diseño experimental (hiperparámetros con notación y justificación — reemplaza a la actual "Parámetros") → 4. Ingeniería de características → 5. Etiquetado (con R3.1 y R7) → 6. Metodología anti-leakage (con R3.2) → 7. Modelos (con mini-marcos R4) → 8. Resultados y caracterización de detectores (ROC/PR/matrices + R5) → 9. Demo → 10. Conclusiones, limitaciones y trabajo futuro → Referencias.

**Aceptación:** los títulos del notebook generado coinciden con esta estructura; ninguna sección tiene más de una idea central.

## 4. Fuera de alcance (explícito)

- Re-entrenamiento sobre grillas de hiperparámetros (θ×k) — trabajo futuro en conclusiones.
- CNN 1D, LSTM, walk-forward — trabajo futuro.
- Backtesting económico / señales de trading.
- Multi-ticker / robustez cruzada entre activos.

## 5. Definition of Done

- [ ] `make build` genera el `.ipynb` desde `build_nb.py` sin warnings de `nbformat.validate`.
- [ ] `make verify` (smoke test con CSV sintético + test de voz R1) pasa en verde.
- [ ] Corrida completa en Colab con el CSV real de RELIANCE: todas las celdas con salida, asserts de alineación OK.
- [ ] Sanity check documentado: sobre datos sintéticos sin señal, ningún modelo supera al baseline (revalidación anti-leakage).
- [ ] `make pdf` genera el PDF `--no-input`; prueba de lectura fría superada (R6).
- [ ] Checklist R4 de defensa oral completada por el autor (cada fórmula explicable sin leer).
- [ ] Guía de defensa actualizada: razonamientos, no machetes (ver [[PLAN]] F5).
- [ ] Entrega realizada antes del 2026-07-30.
