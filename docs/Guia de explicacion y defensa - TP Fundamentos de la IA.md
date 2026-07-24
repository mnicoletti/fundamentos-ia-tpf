

**Trabajo:** Clasificación de dirección de precios: Regresión Logística vs. MLP
**Libro de referencia:** Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press, 2016), Caps. 1–10
**Notebook:** `TP_Fundamentos_IA_Financiero.ipynb`

---

## 1. Elevator pitch (30 segundos)

> "Planteamos un problema de clasificación supervisada sobre datos financieros reales: predecir si el precio de una acción sube, baja o queda lateral a 5 días vista, usando indicadores de análisis técnico como características. Comparamos un modelo lineal (regresión logística) contra una red feedforward (MLP) para responder empíricamente si el problema tiene estructura no lineal aprovechable. El foco del trabajo no es el resultado predictivo — que sabemos cercano al azar en mercados eficientes — sino la **metodología correcta**: split temporal purgado, estandarización sin fuga de información, baseline trivial y métricas robustas al desbalance."

Ese último párrafo es la tesis del TP. Si el profesor se queda con una sola idea, que sea esa.

---

## 2. Mapa notebook → teoría (Goodfellow)

| Sección del notebook | Concepto demostrado | Capítulo |
|---|---|---|
| §1.1 Introducción y problema | Definición formal de aprendizaje: tarea T, desempeño P, experiencia E | Cap. 5.1 |
| §2 Datos y análisis exploratorio | No estacionariedad, colas pesadas, autocorrelación ≈0 de retornos vs. >0 de \|retornos\| | Cap. 5.4 (repr.) |
| §4 Ingeniería de características | Representación de los datos; por qué no usar el precio crudo (no estacionario) | Cap. 5.4 |
| §5 Etiquetado + sensibilidad a θ | Diseño del problema supervisado; clases discretas a partir de variable continua | Cap. 5.1 |
| §6 Metodología anti-leakage (split + purga) | Separación estricta train/test; generalización; i.i.d. como supuesto que acá **no se cumple** y obliga a split temporal | Cap. 5.2–5.3 |
| §7.1 Regresión logística | Modelo lineal probabilístico; softmax; frontera de decisión lineal | Cap. 5.7.1 |
| §7.1 Baseline trivial | Piso de comparación honesto; el error de comparar contra 0% | Cap. 5.2 (contexto de evaluación) |
| §7.2 MLP | Redes feedforward; ReLU; teorema de aproximación universal; backpropagation | Cap. 6 (6.4.1, 6.5) |
| §7.2 Dropout + L2 + early stopping | Regularización: las tres técnicas canónicas del libro | Cap. 7 (7.1, 7.8, 7.12) |
| §7.2 Adam + curvas de pérdida | Optimización basada en gradiente; tasas de aprendizaje adaptativas; diagnóstico de sobreajuste | Cap. 8 (8.5) |
| Softmax + cross-entropy (§7.1) | Máxima verosimilitud; la pérdida como −log P(y\|x); interpretación probabilística de la salida | Cap. 3 (3.13), 5.5 |
| §8.1–8.3 Resultados y análisis por clase | Capacidad del modelo vs. estructura real del problema; underfitting/overfitting; ROC/PR por clase | Cap. 5.2 |
| §9 Demo con gráfico nuevo | Inferencia fuera de muestra; el gráfico como visualización, no como input | — |

---

## 3. Guion de recorrido — 5 minutos por persona

> **Cambio de último momento:** el profesor redujo el tiempo de exposición a **5 minutos por
> persona** (demasiados grupos para la agenda original). Este guion reemplaza al de ~15 min.
> Con 5 minutos no entra explicar arquitecturas, features ni teoría en vivo — eso queda para
> la §4 (preguntas), que ya está armada para responder en el momento si el profesor repregunta.
> El guion en vivo prioriza tres cosas y corta todo lo demás: el problema, el único punto
> metodológico que blinda el trabajo (anti-leakage), y una lectura honesta del resultado.

**0:00–1:00 — Pitch + problema.** El *elevator pitch* de la §1, tal cual (~30 seg). Rematar con
la elección de **3 clases con zona muerta** (±θ=2%) en una frase: sin LATERAL, ruido de ±0.1%
contaría como señal.

**1:00–2:30 — El corazón: anti-leakage, en una sola idea.** "Los datos financieros violan
i.i.d. — la autocorrelación de los retornos es ≈0 pero la de \|retornos\| es >0, está en la
sección de datos —, así que el split es **temporal**, no aleatorio, con una **purga de k=5
muestras** en la frontera para que ninguna etiqueta de entrenamiento mire información del
período de test." No hace falta desglosar el scaler acá salvo que pregunten (queda listo en
§4).

**2:30–4:00 — Resultados y lectura honesta.** Mostrar la tabla comparativa, decir dos frases:
1. "Ni la Regresión Logística (0.443) ni el MLP (0.33–0.45 según la corrida) superaron al
   baseline trivial (0.548) — esperable, coherente con la autocorrelación ≈0 de los datos."
2. "Por F1 macro y AUC, que no premian el sesgo hacia la clase mayoritaria, el MLP superó a la
   Regresión Logística en las cuatro corridas realizadas — ventaja chica pero consistente."

**4:00–4:45 — Cierre.** Una frase: "el resultado modesto es el hallazgo, no un fracaso — confirma
que el problema es tan difícil como predice la teoría de mercados eficientes, y que el pipeline
lo mide sin fuga de información."

**4:45–5:00 — Buffer.** Margen para no pasarse (con 5 min, pasarse 30 seg ya es mucho). Si sobra
tiempo o preguntan por la demo, mencionarla en una frase sin proyectarla: "hay una demo en la
§9 del notebook, sobre una muestra real del período de prueba, si quieren verla."

**Qué se corta respecto del guion largo (y por qué está bien cortarlo):** la explicación
detallada de cada indicador técnico, el desglose de arquitectura del MLP capa por capa, y la
demo en vivo. Ninguna de las tres es la tesis del trabajo — la tesis es la metodología
anti-leakage y la lectura honesta del resultado (§1), y esas dos entran cómodas en 5 minutos.
Si el profesor pregunta por algo cortado, está en §4 con el razonamiento completo.

---

## 4. Preguntas anticipadas del profesor (con razonamiento)

Cada entrada tiene tres partes: la pregunta, la **respuesta** corta (lo que se dice primero), y el **razonamiento** (por qué esa respuesta es cierta — lo que sostiene la respuesta si el profesor repregunta "¿y eso por qué?"). Un machete memorizado tiene la respuesta pero no el razonamiento; con el razonamiento encima, la respuesta se puede reconstruir aunque la repregunta venga de un ángulo distinto al esperado.

**P: ¿Por qué no clasificaron imágenes de los gráficos con una CNN?**
R: Porque la información relevante ya está en forma numérica (los indicadores técnicos), y convertirla en imagen para que una CNN la vuelva a extraer es peor uso de esa información, no mejor.
**Por qué:** el RSI, el MACD, las Bandas de Bollinger son funciones determinísticas del precio — ya *son* la representación extraída a mano de la serie (Cap. 5.4: la elección de representación es en sí un acto de ingeniería). Renderizar esos mismos datos como imagen y pedirle a una CNN que redescubra los patrones visualmente le agrega ruido de renderizado (antialiasing, resolución finita, elección arbitraria de colores/escalas) y le exige aprender de nuevo algo que ya se le entregó explícitamente. Es un argumento general, no solo sobre CNN: cualquier "¿por qué no usaron una arquitectura más compleja acá?" se responde igual — la pregunta correcta no es "¿es más potente?" sino "¿el problema, tal como está planteado, necesita esa potencia?". Acá no: la señal (si la hay) ya vive en 6 indicadores tabulares, no en la forma visual de una curva.

**P: ¿Por qué el accuracy es tan bajo / no supera al baseline?**
R: Porque ni la Regresión Logística (0.443) ni el MLP (entre 0.328 y 0.448 según la corrida) superaron al baseline trivial (0.548) — y eso es el resultado esperado, no una falla del pipeline.
**Por qué:** el baseline trivial (siempre predecir LATERAL, la clase modal con 55% de los casos) es matemáticamente el clasificador óptimo bajo pérdida 0-1 *si no hay señal direccional explotable*. Que ningún modelo real le gane es consistente con lo que la sección de datos ya mostró de forma independiente: la autocorrelación de los retornos es ≈0 en todos los rezagos (evidencia 3 del análisis exploratorio) — es decir, la dirección de mañana no depende de la de hoy de forma medible. Si la dirección no tiene estructura temporal explotable a nivel agregado, tampoco la va a tener un clasificador entrenado sobre esa misma dirección. El resultado del modelo no es una observación aislada: es la confirmación, en la tarea de clasificación, de algo que ya se había predicho antes de entrenar nada. Esa cadena — EDA predice dificultad → resultado confirma la predicción — es más convincente en una defensa que solo decir "el mercado es eficiente".

**P: ¿Cómo saben que no hay data leakage?**
R: Tres mecanismos independientes: split temporal, purga de k muestras en la frontera, y `StandardScaler` ajustado solo con train.
**Por qué cada uno específicamente:** el split temporal evita que información *futura* (en el sentido de fecha calendario) entre al conjunto de entrenamiento — un split aleatorio no distingue "antes" de "después", y el problema es intrínsecamente secuencial (Cap. 5.2, el supuesto i.i.d. no se cumple). La purga ataca una fuga más sutil que el split temporal *simple* no resuelve: la etiqueta de un día se calcula mirando k días hacia adelante, así que las últimas etiquetas de train, aunque estén "antes" en el calendario, contienen información de retornos que ocurren *dentro* del período de test — se verificó explícitamente que el último día que aporta una etiqueta de train es estrictamente anterior al primer día del test, no solo "antes en la tabla". El `StandardScaler` ajustado solo con train evita una fuga de tercer tipo, más sutil todavía: si las medias/desvíos usados para normalizar incorporan estadísticas del test, el modelo indirectamente "sabe" algo sobre la distribución del test antes de evaluarse contra él, incluso sin ver una sola etiqueta. Los tres previenen fugas de naturaleza distinta (fecha, horizonte, estadística) — por eso hacen falta los tres, ninguno cubre lo que cubren los otros dos.

**P: ¿Por qué softmax y cross-entropy y no otra pérdida?**
R: Porque softmax convierte los puntajes del modelo en una distribución de probabilidad válida, y minimizar cross-entropy sobre esa distribución equivale exactamente a maximizar la verosimilitud de las etiquetas observadas.
**Por qué esa equivalencia es cierta (no solo una frase para recitar):** la verosimilitud del dataset completo es el producto de las probabilidades que el modelo asignó a la clase correcta de cada muestra — un número que se vuelve numéricamente inmanejable con miles de muestras (subdesbordamiento). Tomar logaritmo convierte ese producto en una suma (log(a·b) = log a + log b), estable de calcular; invertir el signo convierte "maximizar la log-verosimilitud" en "minimizar el negativo de esa suma" — la misma cantidad, con el signo dado vuelta. Cross-entropy *es* ese negativo promediado sobre las muestras. No es una elección arbitraria entre pérdidas posibles: es la que hace que "entrenar minimizando la pérdida" y "estimar por máxima verosimilitud" sean matemáticamente la misma operación (Cap. 5.5, y el bloque `Concepto — Cross-entropy` del propio notebook, con el ejemplo numérico completo).

**P: ¿Qué hace exactamente el dropout y por qué 0.3?**
R: En cada paso de entrenamiento apaga al azar el 30% de las neuronas de la capa, forzando que ninguna se vuelva indispensable.
**Por qué funciona como regularizador:** si una neurona puede faltar en el próximo lote, la red no puede depender de que "la neurona 47 siempre detecte tal patrón" — tiene que repartir esa capacidad de forma redundante entre varias neuronas. Esa redundancia es justamente lo opuesto a memorizar: memorizar es depender de combinaciones muy específicas de pesos que ajustan una muestra particular del train; la redundancia forzada hace que esas combinaciones específicas sean frágiles (se rompen si falta una neurona) y por lo tanto el entrenamiento las evita. El 0.3 es un valor estándar para capas densas de este tamaño — no se buscó por grilla porque el foco del TP es la comparación metodológica LogReg-vs-MLP, no la maximización de una métrica (si un profesor pregunta "¿probaron con 0.5?", la respuesta honesta es que no, y por qué no: hacerlo sería optimizar hacia el conjunto de test, exactamente el tipo de fuga sutil que el resto del trabajo se cuidó de evitar).

**P: ¿Por qué Adam y no SGD puro?**
R: Porque Adam le da a cada peso una tasa de aprendizaje efectiva propia, ajustada según el historial reciente de su gradiente, en vez de una tasa única para toda la red.
**Por qué eso importa en este problema en particular:** con datos financieros ruidosos y relativamente pocas muestras de entrenamiento (2157 en la corrida canónica), los gradientes de distintos pesos se comportan de forma muy despareja — algunos con señal consistente, otros dominados por ruido. Con SGD y una tasa única, hay que elegir un compromiso: una tasa que sirva para los pesos "ruidosos" (chica, para no sobrecorregir) penaliza a los pesos con señal clara (avanzan innecesariamente despacio), y viceversa. Adam evita ese compromiso dándole a cada peso su propio paso efectivo. El costo es que SGD puro con una tasa bien calibrada a mano puede llegar a resultados comparables — Adam no es estrictamente necesario, es una elección de practicidad de ingeniería, no un requisito teórico.

**P: ¿Las clases están balanceadas? ¿Cómo lo manejan?**
R: No: LATERAL es el 55% del test, BAJA y SUBE se reparten el resto de forma desigual. Se maneja con F1 macro como métrica principal y `class_weight` balanceado en el entrenamiento del MLP.
**Por qué ambas medidas son necesarias y no redundantes:** `class_weight` actúa *durante el entrenamiento* — le dice a la función de pérdida que un error en una clase minoritaria pesa más que un error en la mayoritaria, para que el optimizador no tenga incentivo a "ignorar" las clases chicas. F1 macro actúa *después*, en la evaluación — promedia el F1 de cada clase con el mismo peso, así que un modelo que solo acierta en LATERAL (como efectivamente le pasa a la Regresión Logística, recall 0.06 en BAJA) queda expuesto en F1 macro aunque su accuracy sea alta. Sin F1 macro, la tabla de resultados escondería exactamente el sesgo que `class_weight` intenta corregir en el entrenamiento — una sin la otra deja un hueco.

**P: El MLP le gana a la Regresión Logística, ¿sí o no?**
R: Depende de la métrica: por accuracy, la Regresión Logística gana en la mayoría de las corridas (0.443 contra 0.328–0.448 del MLP); por F1 macro, AUC macro y mAP, el MLP ganó en las cuatro corridas realizadas, sin excepción.
**Por qué la respuesta correcta es "depende" y no un número:** la accuracy de la Regresión Logística está inflada por su sesgo hacia LATERAL — predice esa clase el 77% de las veces (mucho más que su 55% de prevalencia real), así que acierta "gratis" en una fracción grande del test sin haber aprendido nada direccional. Las métricas robustas al desbalance (F1 macro, AUC, AP) no premian ese atajo, porque promedian por clase en vez de por muestra. Que el MLP gane consistentemente ahí, pero por un margen modesto (2 a 6 puntos porcentuales), es evidencia de una ventaja real pero chica del modelo no lineal — ni "el MLP no sirve" ni "el MLP es claramente mejor" son lecturas honestas; la lectura honesta es la que distingue qué mide cada métrica y por qué difieren.

**P: ¿Corrieron el modelo una sola vez? ¿Da siempre el mismo resultado?**
R: No: se corrió cuatro veces con el mismo código y los mismos datos, y el MLP dio accuracy distinta cada vez (0.333, 0.394, 0.448, 0.328) — la Regresión Logística, en cambio, dio exactamente 0.443 en las cuatro.
**Por qué pasa esto y por qué no se "arregló":** `LogisticRegression(random_state=SEED)` es una optimización convexa determinística — el mismo problema siempre tiene el mismo óptimo, encontrado por el mismo camino. El entrenamiento del MLP con Keras/TensorFlow no lo es del todo, ni siquiera fijando `np.random.seed`, `tf.random.set_seed` y `tf.config.experimental.enable_op_determinism()` (se probaron los tres): hay operaciones internas de TensorFlow —paralelismo en reducciones numéricas, orden de ejecución en CPU— que esas semillas no terminan de fijar en todas las versiones de la librería. En vez de perseguir un número exacto (lo que se intentó, y no funcionó con la mitigación estándar), el análisis de resultados (R5, notebook) reporta el *rango* observado entre corridas y confía en los patrones que se sostuvieron las cuatro veces, no en un valor puntual. Esto, lejos de ser una debilidad para ocultar, es un punto metodológico genuino: la inestabilidad del propio entrenamiento es evidencia adicional de que la señal disponible es débil — si hubiera estructura fuerte para aprender, corridas repetidas deberían converger a soluciones parecidas.

**P: ¿Esto sirve para operar en el mercado?**
R: No, y el TP lo dice explícitamente.
**Por qué no, en concreto:** faltan al menos cuatro cosas que separan "un clasificador con AUC ligeramente por encima de 0.5" de "una estrategia operable": costos de transacción y slippage (que en un horizonte de 5 días pueden superar cualquier ventaja estadística de este tamaño), validación *walk-forward* con reentrenamientos sucesivos (acá se entrena una sola vez sobre un split fijo, no se prueba si la ventaja se sostiene rolling en el tiempo), un tamaño de efecto económicamente significativo (una mejora de 0.03 en AUC no dice nada sobre si es rentable después de costos), y robustez multi-activo (un solo ticker, RELIANCE, no generaliza). El objetivo del trabajo es demostrar fundamentos de ML con metodología correcta, no proponer una señal de trading — la brecha entre esas dos cosas es justamente la lista anterior.

**P: ¿Qué agregarían con más tiempo?**
R: Validación *walk-forward*, CNN 1D o LSTM para comparar contra arquitecturas con más estructura temporal, y — a diferencia de lo que se pensaba al principio del TP — ya no falta el análisis de sensibilidad al umbral θ: eso se agregó (R7, sección de etiquetado).
**Por qué esas y no otras extensiones:** *walk-forward* ataca la limitación más importante que queda (un solo split, entrenado una vez) y es el estándar de la industria para series financieras (López de Prado, Cap. 7) — es la extensión con mayor retorno metodológico por esfuerzo. CNN 1D es atractiva porque una media móvil *es*, literalmente, una convolución con un kernel uniforme — el argumento de que la arquitectura tiene sentido para este dominio no es solo "es más moderna", es que el propio feature engineering manual (SMA, EMA) ya está haciendo, a mano, una operación que una CNN podría aprender a hacer mejor. LSTM serviría para dependencias más largas que las que captura la ventana fija de w=30 días. La sensibilidad a k (horizonte) es la variante que efectivamente *no* se hizo y sigue pendiente — por consistencia, quedaría mencionada como trabajo futuro real, no aspiracional.

---

## 5. Conceptos clave para tener frescos (glosario de defensa)

- **Capacidad** (Cap. 5.2): habilidad del modelo para ajustar variedad de funciones. Poca → underfitting; mucha sin regularización → overfitting.
- **Teorema de aproximación universal** (Cap. 6.4.1): una capa oculta con suficientes unidades y activación no lineal aproxima cualquier función continua en un compacto. No garantiza que el entrenamiento la encuentre.
- **Backpropagation** (Cap. 6.5): aplicación de la regla de la cadena para computar gradientes de la pérdida respecto de todos los pesos, en tiempo lineal en el tamaño de la red.
- **Regularización** (Cap. 7): toda modificación destinada a reducir el error de generalización sin necesariamente reducir el de entrenamiento. En el TP: L2 (weight decay, 7.1), early stopping (7.8), dropout (7.12).
- **Early stopping como regularizador** (Cap. 7.8): Goodfellow lo describe como el regularizador "gratis" más usado; restringe implícitamente el espacio de parámetros alcanzable.
- **Máxima verosimilitud** (Cap. 5.5): cross-entropy = negative log-likelihood; entrenar es hacer inferencia estadística.
- **i.i.d. y por qué falla acá** (Cap. 5.2): los supuestos de muestras independientes e idénticamente distribuidas no valen en series temporales; de ahí el split temporal y la purga.

---

## 6. Checklist R4 — explicabilidad en voz alta

Validado el 2026-07-23: los 8 bloques `> **Concepto — ...**` embebidos en el notebook (sección
"7. Modelos") se revisaron uno por uno — fórmula con ejemplo numérico primero, confirmación de
poder explicarla en voz alta sin mirarla después (PLAN.md F3, "Loop de validación por bloque").

| # | Concepto | Ubicación en el notebook | Explicable en voz alta |
|---|---|---|---|
| 1 | Softmax como distribución de probabilidad | §7.1, tras la fórmula de la regresión logística | ✅ |
| 2 | Cross-entropy como negative log-likelihood | §7.1, tras el entrenamiento de la regresión logística | ✅ (requirió una segunda pasada, con ejemplo extendido a 4 muestras) |
| 3 | Capacidad y under/overfitting | §7.2, antes de `construir_mlp` | ✅ |
| 4 | Aproximación universal ("representar" ≠ "aprender") | §7.2, antes de `construir_mlp` | ✅ |
| 5 | Backpropagation como regla de la cadena | §7.2, tras `construir_mlp` | ✅ |
| 6 | Regularización de parámetros: L2 + Dropout | §7.2, tras `construir_mlp` | ✅ |
| 7 | Adam (tasas de aprendizaje adaptativas) | §7.2, tras `construir_mlp` | ✅ |
| 8 | Early stopping | §7.2, epígrafe de la figura de curvas de pérdida | ✅ |

**8/8 — checklist F3 completa.**

Nota sobre el agrupamiento: SPEC.md lista "L2 / dropout / early stopping" como un solo bullet
de conceptos mínimos, pero la aceptación de R4 pide "8 conceptos" — una inconsistencia real del
propio documento (no corregida en SPEC.md/PLAN.md, solo resuelta acá). Se agrupó L2+Dropout en
un solo bloque ("Regularización de parámetros") y se dejó *early stopping* aparte, porque ya
contaba con su propia figura de anclaje (curvas de pérdida anotadas, R3.3).

Pendiente para F5: notas al pie en el PDF con los conceptos de base que estos 8 bloques ya dan
por sabidos (qué es softmax/regresión logística/MLP en una línea), pedido por Max durante la
revisión de este checklist.

---

## 7. Anexo — Registro de revisión técnica (evaluación previa a la entrega)

El notebook fue verificado ejecutando el pipeline completo de punta a punta con datos sintéticos (paseo aleatorio geométrico) y chequeos programáticos de alineación. Hallazgos y correcciones aplicadas:

| # | Hallazgo | Severidad | Corrección |
|---|---|---|---|
| 1 | `LogisticRegression(multi_class="multinomial")` — parámetro eliminado en scikit-learn ≥ 1.7, produce `TypeError` | Crítica | Eliminado; el solver por defecto (lbfgs) ya es multinomial |
| 2 | Fuga de horizonte en la frontera train/test: las últimas 5 etiquetas de train miraban días del período de test | Metodológica | Purga (*purged split*) de `HORIZONTE` muestras; verificado sin solape |
| 3 | *Off-by-one* en el mapeo de índices de la demo (mostraba el día equivocado) | Funcional | Fórmula corregida `VENTANA + (n_train + idx_local) − 1` + asserts de alineación en el propio notebook |
| 4 | Posible `KeyError` al graficar muestras cercanas al final del test | Borde | Clamp `fin = min(idx + HORIZONTE, len(d)−1)` |

Verificación de sanidad adicional: sobre datos sintéticos sin señal (martingala), ningún modelo superó al baseline trivial — comportamiento esperado que valida la ausencia de *leakage* en el pipeline.

---

*Referencias: Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. — López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.*
