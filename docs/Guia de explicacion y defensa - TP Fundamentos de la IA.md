

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
| §1.1 Marco conceptual | Definición formal de aprendizaje: tarea T, desempeño P, experiencia E | Cap. 5.1 |
| §5 Ingeniería de características | Representación de los datos; por qué no usar el precio crudo (no estacionario) | Cap. 5.4 |
| §6 Etiquetado con umbral | Diseño del problema supervisado; clases discretas a partir de variable continua | Cap. 5.1 |
| §7 Split temporal + purga | Separación estricta train/test; generalización; i.i.d. como supuesto que acá **no se cumple** y obliga a split temporal | Cap. 5.2–5.3 |
| §8 Regresión logística | Modelo lineal probabilístico; softmax; frontera de decisión lineal | Cap. 5.7.1 |
| §8 Baseline trivial | Piso de comparación honesto; el error de comparar contra 0% | Cap. 5.2 (contexto de evaluación) |
| §9 MLP | Redes feedforward; ReLU; teorema de aproximación universal; backpropagation | Cap. 6 (6.4.1, 6.5) |
| §9 Dropout + L2 + early stopping | Regularización: las tres técnicas canónicas del libro | Cap. 7 (7.1, 7.8, 7.12) |
| §9 Adam + curvas de pérdida | Optimización basada en gradiente; tasas de aprendizaje adaptativas; diagnóstico de sobreajuste | Cap. 8 (8.5) |
| Softmax + cross-entropy | Máxima verosimilitud; la pérdida como −log P(y\|x); interpretación probabilística de la salida | Cap. 3 (3.13), 5.5 |
| §10 Comparativa | Capacidad del modelo vs. estructura real del problema; underfitting/overfitting | Cap. 5.2 |
| §11 Demo con gráfico nuevo | Inferencia fuera de muestra; el gráfico como visualización, no como input | — |

---

## 3. Guion de recorrido sugerido (defensa de ~15 min)

**Minutos 0–2 — El problema.** Presentar la tarea T/P/E (tabla de §1.1). Enfatizar la elección de **3 clases con zona muerta** (umbral ±2%): sin la clase LATERAL, movimientos de ruido de ±0.1% contarían como señal, contaminando el aprendizaje.

**Minutos 2–5 — Los datos y las features.** Justificar cada indicador como *feature engineering* informado por dominio: el RSI y el MACD son transformaciones no lineales del precio que un analista humano usa; se las damos al modelo pre-computadas en lugar de esperar que las descubra. Conexión directa con Goodfellow 5.4: "la elección de la representación tiene un efecto enorme en el desempeño".

**Minutos 5–8 — La metodología anti-leakage (el corazón).** Explicar las tres capas de protección:
1. **Split temporal**, no aleatorio: los datos financieros violan el supuesto i.i.d. (Cap. 5.2); un shuffle mete el futuro en el entrenamiento.
2. **Purga de HORIZONTE muestras** en la frontera: las últimas etiquetas de train se calculan mirando días que pertenecen al test. Referencia externa: López de Prado, *Advances in Financial ML* (2018), Cap. 7 (*purged K-fold*).
3. **StandardScaler ajustado solo con train**: las medias/desvíos del test no deben influir en la transformación.

**Minutos 8–11 — Los dos modelos.** Regresión logística = softmax sobre combinación lineal; frontera = hiperplano. MLP = dos capas ocultas ReLU; por el teorema de aproximación universal (6.4.1) puede representar fronteras arbitrarias — pero "poder representar" no es "poder aprender", distinción que el propio Goodfellow subraya. Mostrar las curvas de pérdida y señalar dónde actuó el early stopping.

**Minutos 11–14 — Resultados y lectura honesta.** Presentar la tabla comparativa contra el baseline trivial. Si el MLP mejora poco (esperable): explicar que la señal predictiva en retornos de corto plazo es mínima por cuasi-eficiencia del mercado; la capacidad extra del MLP solo ayuda si hay estructura que capturar (Cap. 5.2). **Este resultado modesto es el hallazgo, no un fracaso.**

**Minuto 14–15 — Demo.** El gráfico nuevo del período de test con las predicciones de ambos modelos lado a lado. Remarcar: el gráfico es visualización del resultado; el modelo consume el tensor numérico de indicadores.

---

## 4. Preguntas anticipadas del profesor (con respuesta)

**P: ¿Por qué no clasificaron imágenes de los gráficos con una CNN?**
R: Renderizar la serie a píxeles destruye la estructura numérica que ya tenemos y agrega ruido de renderizado (ejes, colores). La CNN tendría que reconstruir desde la imagen la información que ya poseemos en forma tensorial. Es metodológicamente circular. Además, CNN corresponde al Cap. 9, fuera del alcance elegido (Caps. 5–8).

**P: ¿Por qué el accuracy es tan bajo / cercano al baseline?**
R: Porque el problema es intrínsecamente difícil: bajo la hipótesis de mercados cuasi-eficientes, los retornos de corto plazo se comportan casi como una martingala. Verificamos el pipeline con datos sintéticos de paseo aleatorio puro y, correctamente, ningún modelo superó al baseline trivial — el pipeline honesto no encuentra señal donde no la hay. Con datos reales, cualquier mejora sobre el baseline, aunque pequeña, indica estructura residual. Un accuracy alto acá sería sospecha de *leakage*, no mérito.

**P: ¿Cómo saben que no hay data leakage?**
R: Tres mecanismos: split temporal, purga de HORIZONTE muestras en la frontera (verificada: el último día que "mira" una etiqueta de train es estrictamente anterior al primer día etiquetado del test), y scaler ajustado solo con train. Además la validación interna de Keras (`validation_split`) toma el tramo final del train sin mezclar, preservando el orden temporal.

**P: ¿Por qué softmax y cross-entropy y no otra pérdida?**
R: La salida softmax define una distribución de probabilidad sobre las 3 clases; minimizar la entropía cruzada equivale a maximizar la log-verosimilitud de las etiquetas observadas (Cap. 5.5). Es el estimador de máxima verosimilitud del modelo, con propiedades estadísticas bien caracterizadas (consistencia, eficiencia asintótica, Cap. 5.5.2).

**P: ¿Qué hace exactamente el dropout y por qué 0.3?**
R: En cada paso de entrenamiento apaga aleatoriamente el 30% de las activaciones, forzando redundancia y evitando co-adaptación de neuronas; se interpreta como un ensamble implícito de subredes (Cap. 7.12). El valor 0.3 es un hiperparámetro estándar para capas densas medianas; el TP no hace búsqueda exhaustiva de hiperparámetros porque el foco es comparativo, no de maximización.

**P: ¿Por qué Adam y no SGD puro?**
R: Adam adapta la tasa de aprendizaje por parámetro usando momentos de primer y segundo orden del gradiente (Cap. 8.5.3), lo que acelera la convergencia en problemas con gradientes ruidosos como éste. SGD puro funcionaría con más tuning de learning rate; la comparación SGD vs. Adam es una extensión natural mencionada en las conclusiones.

**P: ¿Las clases están balanceadas? ¿Cómo lo manejan?**
R: No perfectamente; por eso (a) reportamos F1 macro además de accuracy, y (b) entrenamos el MLP con `class_weight` balanceado, que pondera la pérdida inversamente a la frecuencia de cada clase.

**P: ¿Esto sirve para operar en el mercado?**
R: No, y el TP lo dice explícitamente. Faltarían costos de transacción, slippage, validación walk-forward con múltiples ventanas, y un tamaño de efecto económicamente significativo. El objetivo es demostrar fundamentos de ML con metodología correcta.

**P: ¿Qué agregarían con más tiempo?**
R: CNN 1D (Cap. 9) para explotar localidad temporal — con el argumento elegante de que una media móvil es literalmente una convolución — y LSTM (Cap. 10) para dependencias largas. También validación walk-forward y análisis de sensibilidad al umbral θ y al horizonte k.

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

## 6. Anexo — Registro de revisión técnica (evaluación previa a la entrega)

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
