# CLAUDE.md — TP Fundamentos de la IA (Maestría en IA, UP)

## Qué es este proyecto

Trabajo práctico académico: clasificación de dirección de precios (SUBE/LATERAL/BAJA a 5 días) comparando Regresión Logística vs. MLP, sobre el dataset Kaggle Nifty 100 resampleado a diario. **La tesis del trabajo es metodológica, no predictiva**: split temporal purgado, sin data leakage, baseline honesto. Un accuracy alto acá sería síntoma de bug, no de éxito.

Entregables: notebook `.ipynb` ejecutado (Colab) + PDF de exposición sin código. Deadline duro: **2026-07-30**.

Documentos rectores — leelos antes de tocar código:
- `SPEC.md` — requerimientos R1–R8 con criterios de aceptación. **Las decisiones D1–D8 están cerradas: no las re-litigues ni las "mejores".**
- `PLAN.md` — fases F0–F5 con gates de salida.
- `docs/adr/` — ADR-001 a 005. Si una tarea te tienta a contradecir un ADR, frenás y preguntás.

## Regla de oro: fuente de verdad

**`build_nb.py` es la única fuente de verdad. El `.ipynb` es un artefacto generado. NUNCA edites un `.ipynb` directamente** (ADR-001). Flujo: editar `build_nb.py` → `make build` → `make verify`.

## Estructura del repo

```
tp-fundamentos-ia/
├── CLAUDE.md              # este archivo
├── SPEC.md                # requerimientos y decisiones cerradas
├── PLAN.md                # fases y gates
├── docs/adr/              # ADR-001..005 (Nygard+MADR, wikilinks Obsidian)
├── build_nb.py            # FUENTE DE VERDAD del notebook
├── verify_pipeline.py     # smoke test + test_voz (ver abajo)
├── Makefile               # build / verify / pdf / clean
├── data/                  # CSVs Kaggle (gitignored; RELIANCE canónico, ADR-005)
└── dist/                  # .ipynb y .pdf generados (gitignored salvo entrega congelada)
```

## Comandos

```bash
make build     # build_nb.py -> dist/TP_Fundamentos_IA_Financiero.ipynb (valida con nbformat)
make verify    # depende de build. Corre verify_pipeline.py (ver abajo). SIEMPRE antes de commitear.
make pdf       # nbconvert --to webpdf --no-input, excluye celdas con tag 'no-pdf' (ADR-004)
               # fallback si webpdf falla: make pdf-html (html + instrucción de print)
make clean     # limpia dist/
```

`verify_pipeline.py` hace, en orden:
1. Genera un CSV sintético 5-min formato Kaggle (GBM, paseo aleatorio).
2. Ejecuta las celdas de código del notebook generado en orden, con `yfinance` y Keras/TensorFlow **mockeados** (el MLP se reemplaza por probabilidades softmax aleatorias) — el pipeline completo debe correr sin red y sin GPU.
3. Verifica los asserts de alineación de índices y la purga sin solape.
4. `test_voz` (R1): grep de términos prohibidos sobre las fuentes markdown de `build_nb.py` — "instal", "Colab", "celda", "ejecut", "cambiar el", "permite experimentar". Las celdas operativas legítimas deben llevar tag `no-pdf` y quedan exentas.
5. Sanity semántico: sobre el CSV sintético (sin señal), ningún modelo debe superar al baseline trivial por margen significativo. Si lo supera, hay leakage: **es un bug, no una mejora**.

## Convenciones de código y prosa

- Python 3.11+. En scripts bash: `set -euo pipefail`.
- Prosa del notebook: registro académico en español, impersonal o primera persona plural. **Prohibido**: voz de asistente ("instalamos", "cambiar X permite experimentar"), referencias operativas (Colab, celdas, ejecutar), referencias al código desde la prosa ("como se ve arriba") — el PDF sale sin código.
- Toda figura lleva epígrafe interpretativo de ≥2 oraciones: qué se ve y qué se concluye.
- Mini-marcos teóricos (R4): bloque `> **Concepto — [nombre]**` con estructura fija: intuición llana → fórmula LaTeX → término por término → anclaje en figura propia → cita a Goodfellow. **El autor está aprendiendo la materia con este documento**: escribí para un lector que parte de cero. Nunca uses una fórmula sin explicar cada símbolo.
- Notación consistente en todo el documento: θ (umbral), k (horizonte), w (ventana). Si introducís un símbolo nuevo, definilo en su primera aparición.
- Hiperparámetros del experimento (D8): θ=2%, k=5, w=30, split 80/20, purga=k. No los cambies.
- Celdas markdown en `build_nb.py`: strings raw (`r"""..."""`) para que LaTeX no rompa escapes.

## Qué NO hacer

- No edites `.ipynb` (ADR-001). No cambies decisiones D1–D8 (SPEC §2). No agregues re-entrenamientos ni grillas de hiperparámetros (fuera de alcance, SPEC §4). No elijas otro ticker por resultados (ADR-005: sería cherry-picking; el suplente HDFCBANK se activa solo por calidad de datos).
- No escribas números de resultados concretos en la prosa antes de F4 (la corrida canónica): toda afirmación cuantitativa pre-F4 es un placeholder que va a quedar mal. Redactá en términos cualitativos condicionales.
- No "mejores" el modelo para subir métricas. La honestidad metodológica ES el entregable.
- No inventes citas: las únicas referencias son Goodfellow et al. (2016) caps. 1–10 y López de Prado (2018) cap. 7.

## Definition of Done por tarea

Una tarea está terminada cuando: `make build && make verify` en verde, el criterio de aceptación del requerimiento correspondiente en `SPEC.md` §3 se cumple textualmente, y el diff de `build_nb.py` es revisable (cambios de prosa separados de cambios de código en commits distintos cuando sea posible).

## Contexto del autor (para calibrar explicaciones)

CTO con background DevOps/SRE fuerte: pipeline, testing, IaC y rigor de ingeniería no requieren explicación. La teoría de ML de la materia (probabilidad, optimización, teoría de aprendizaje) sí: es el área que este TP debe enseñarle. Ante la duda de si algo teórico "es obvio": no lo es, explicalo.
