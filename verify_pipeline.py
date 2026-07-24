#!/usr/bin/env python3
r"""Smoke test del notebook generado por build_nb.py (F0/F1).

Verifica, sin red y sin GPU, que el pipeline completo corre de punta a punta:

1. test_voz (R1): grep de términos prohibidos sobre las celdas markdown
   declaradas en build_nb.py, exceptuando las que llevan tag `no-pdf`.
2. Genera un CSV sintético formato Kaggle Nifty (velas de 5 minutos, paseo
   aleatorio geométrico sin deriva -> sin señal direccional real).
3. Ejecuta las celdas de código del notebook generado, en orden, con
   `yfinance` y `tensorflow`/`keras` mockeados (el MLP se reemplaza por
   probabilidades softmax aleatorias).
4. Verifica que la purga train/test no deja solapamiento (además de los
   asserts de alineación de índices que ya trae el propio notebook).
5. Sanity check anti-leakage: sobre datos sintéticos sin señal, ningún
   modelo debe superar al baseline trivial por un margen significativo. Si
   lo supera, hay un bug de leakage, no una mejora.

Uso:
    make verify                        # build + verify encadenados
    python3 verify_pipeline.py         # asume dist/TP_Fundamentos_IA_Financiero.ipynb ya generado
    python3 verify_pipeline.py --help
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import nbformat

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_NOTEBOOK = REPO_ROOT / "dist" / "TP_Fundamentos_IA_Financiero.ipynb"
SEED = 42

# Lista exacta de SPEC.md R1 / CLAUDE.md: referencias operativas u "voz de
# asistente" prohibidas en la prosa académica del notebook.
PROHIBIDOS_VOZ = ["instal", "colab", "celda", "ejecut", "cambiar el", "permite experimentar"]


# ---------------------------------------------------------------------------
# 0. test_voz (R1): sin voz de asistente en las celdas markdown no exentas
# ---------------------------------------------------------------------------

def test_voz() -> None:
    """Ninguna celda markdown no exenta (`no-pdf`) debe contener voz de asistente.

    Opera sobre las celdas tal como las declara build_nb.py, no sobre el
    .ipynb ya construido: así una celda etiquetada `no-pdf` (nota operativa
    legítima, SPEC R1) queda exenta sin necesidad de duplicar esa lógica acá.
    """
    import build_nb

    violaciones = []
    for celda in build_nb.construir_celdas():
        if celda["cell_type"] != "markdown":
            continue
        if "no-pdf" in celda.get("metadata", {}).get("tags", []):
            continue
        texto = celda["source"].lower()
        for termino in PROHIBIDOS_VOZ:
            if termino in texto:
                violaciones.append(f"{termino!r} en: {celda['source'].strip()[:70]!r}...")

    assert not violaciones, "voz de asistente detectada (R1):\n  " + "\n  ".join(violaciones)


# ---------------------------------------------------------------------------
# 1. CSV sintético formato Kaggle Nifty (5 minutos, paseo aleatorio geométrico)
# ---------------------------------------------------------------------------

def generar_csv_sintetico(destino: Path, dias: int = 250, precio_inicial: float = 500.0) -> Path:
    """Genera un CSV con el esquema del dataset Kaggle Nifty 100 (5-min).

    El precio sigue un movimiento browniano geométrico sin deriva: por
    construcción no tiene autocorrelación ni señal direccional real. Esa
    propiedad es la que explota `verificar_sanity_anti_leakage`.
    """
    rng = np.random.default_rng(SEED)
    barras_por_dia = 78  # sesión de 6.5h a barras de 5 min (09:15-15:30, estilo NSE)

    dias_habiles = pd.bdate_range("2015-01-01", periods=dias)
    timestamps = []
    for dia in dias_habiles:
        inicio = dia + pd.Timedelta(hours=9, minutes=15)
        timestamps.extend(pd.date_range(inicio, periods=barras_por_dia, freq="5min"))
    timestamps = pd.DatetimeIndex(timestamps)
    n = len(timestamps)

    sigma = 0.0015
    log_ret = rng.normal(0.0, sigma, size=n)
    close = precio_inicial * np.exp(np.cumsum(log_ret))

    ruido = np.abs(rng.normal(0.0, sigma / 2, size=n))
    open_ = close * (1 + rng.normal(0.0, sigma / 3, size=n))
    high = np.maximum(open_, close) * (1 + ruido)
    low = np.minimum(open_, close) * (1 - ruido)
    volume = rng.integers(1_000, 50_000, size=n)

    df = pd.DataFrame({
        "date": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destino, index=False)
    return destino


# ---------------------------------------------------------------------------
# 2. Mocks: yfinance y TensorFlow/Keras (sin red, sin GPU)
# ---------------------------------------------------------------------------

def _instalar_mock_yfinance() -> None:
    modulo = types.ModuleType("yfinance")

    def download(ticker, start=None, end=None, auto_adjust=True, **kwargs):
        rng = np.random.default_rng(SEED)
        n = 500
        idx = pd.bdate_range(start=start or "2015-01-01", periods=n)
        close = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
        return pd.DataFrame(
            {"Open": close, "High": close, "Low": close, "Close": close,
             "Volume": rng.integers(1_000, 10_000, n)},
            index=idx,
        )

    modulo.download = download
    sys.modules["yfinance"] = modulo


class _MockLayer:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _MockDense(_MockLayer):
    @property
    def units(self):
        return self.args[0] if self.args else self.kwargs.get("units")


class _MockHistory:
    def __init__(self, history: dict):
        self.history = history


class _MockSequential:
    """Reemplazo de keras.Sequential: no entrena, produce softmax aleatoria.

    Lo que valida el smoke test es la integridad del pipeline (alineación de
    índices, purga, ausencia de leakage), no el entrenamiento real de la red
    — por eso el MLP se mockea en vez de correr TensorFlow real.
    """

    def __init__(self, capas):
        densas = [c for c in capas if isinstance(c, _MockDense)]
        self.n_clases = densas[-1].units if densas else 3
        self._rng = np.random.default_rng(SEED)

    def compile(self, **kwargs):
        pass

    def summary(self):
        print(f"[mock] Sequential -> softmax de {self.n_clases} clases")

    def fit(self, X, y, epochs=1, callbacks=None, **kwargs):
        patience = 15
        for cb in (callbacks or []):
            patience = getattr(cb, "patience", patience)
        n_epocas = min(epochs, max(patience, 5))
        loss = np.linspace(1.1, 0.7, n_epocas) + self._rng.normal(0, 0.02, n_epocas)
        val_loss = np.linspace(1.1, 0.85, n_epocas) + self._rng.normal(0, 0.03, n_epocas)
        acc = np.linspace(0.35, 0.55, n_epocas) + self._rng.normal(0, 0.01, n_epocas)
        val_acc = np.linspace(0.34, 0.45, n_epocas) + self._rng.normal(0, 0.01, n_epocas)
        return _MockHistory({
            "loss": loss.tolist(), "val_loss": val_loss.tolist(),
            "accuracy": acc.tolist(), "val_accuracy": val_acc.tolist(),
        })

    def predict(self, X, verbose=0):
        logits = self._rng.normal(size=(len(X), self.n_clases))
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)


class _MockEarlyStopping:
    def __init__(self, monitor="val_loss", patience=0, restore_best_weights=False):
        self.monitor = monitor
        self.patience = patience
        self.restore_best_weights = restore_best_weights


class _MockAdam:
    def __init__(self, *args, **kwargs):
        pass


def _instalar_mock_tensorflow() -> None:
    tf_mod = types.ModuleType("tensorflow")
    tf_mod.__path__ = []

    keras_mod = types.ModuleType("tensorflow.keras")
    layers_mod = types.ModuleType("tensorflow.keras.layers")
    optimizers_mod = types.ModuleType("tensorflow.keras.optimizers")
    callbacks_mod = types.ModuleType("tensorflow.keras.callbacks")
    regularizers_mod = types.ModuleType("tensorflow.keras.regularizers")

    layers_mod.Input = lambda *a, **kw: _MockLayer(*a, **kw)
    layers_mod.Dense = _MockDense
    layers_mod.Dropout = lambda *a, **kw: _MockLayer(*a, **kw)
    optimizers_mod.Adam = _MockAdam
    callbacks_mod.EarlyStopping = _MockEarlyStopping
    regularizers_mod.l2 = lambda *a, **kw: None

    keras_mod.Sequential = _MockSequential
    keras_mod.layers = layers_mod
    keras_mod.optimizers = optimizers_mod
    keras_mod.callbacks = callbacks_mod
    keras_mod.regularizers = regularizers_mod

    class _Random:
        @staticmethod
        def set_seed(seed):
            pass

    tf_mod.random = _Random()
    tf_mod.keras = keras_mod
    tf_mod.__version__ = "mocked"

    sys.modules["tensorflow"] = tf_mod
    sys.modules["tensorflow.keras"] = keras_mod
    sys.modules["tensorflow.keras.layers"] = layers_mod
    sys.modules["tensorflow.keras.optimizers"] = optimizers_mod
    sys.modules["tensorflow.keras.callbacks"] = callbacks_mod
    sys.modules["tensorflow.keras.regularizers"] = regularizers_mod


# ---------------------------------------------------------------------------
# 3. Ejecución de las celdas de código del notebook generado
# ---------------------------------------------------------------------------

_CSV_NIFTY_RE = re.compile(r'CSV_NIFTY\s*=\s*".*?"')


def ejecutar_notebook(notebook_path: Path, csv_sintetico: Path) -> dict:
    nb = nbformat.read(notebook_path, as_version=4)

    _instalar_mock_yfinance()
    _instalar_mock_tensorflow()

    namespace: dict = {}
    for celda in nb.cells:
        if celda.cell_type != "code":
            continue
        fuente = celda.source
        if fuente.lstrip().startswith("!"):
            continue  # magics de shell (ej. !pip install): no aplican fuera de Colab
        if "CSV_NIFTY" in fuente:
            fuente = _CSV_NIFTY_RE.sub(f'CSV_NIFTY   = "{csv_sintetico}"', fuente)
        exec(compile(fuente, "<celda>", "exec"), namespace)
    return namespace


# ---------------------------------------------------------------------------
# 4. Asserts de purga y sanity check anti-leakage
# ---------------------------------------------------------------------------

def verificar_purga_sin_solape(ns: dict) -> None:
    horizonte, purga = ns["HORIZONTE"], ns["PURGA"]
    n_train = ns["n_train"]
    x_train, x_test, x_flat = ns["X_train"], ns["X_test"], ns["X_flat"]

    assert purga == horizonte, f"la purga ({purga}) debe ser exactamente el horizonte ({horizonte})"
    assert x_train.shape[0] == n_train - purga, "el train no descarta las últimas PURGA muestras"
    assert x_train.shape[0] + purga + x_test.shape[0] == x_flat.shape[0], (
        "train + purga + test no reconstruye el total de ventanas: hay solape o pérdida de datos"
    )


def verificar_sanity_anti_leakage(ns: dict, margen: float = 0.15) -> None:
    acc_trivial = ns["acc_trivial"]
    for nombre, acc in (("Regresión Logística", ns["acc_lr"]), ("MLP", ns["acc_mlp"])):
        assert acc <= acc_trivial + margen, (
            f"{nombre} supera al baseline trivial por más de {margen:.0%} sobre datos "
            f"sintéticos sin señal (accuracy={acc:.3f}, baseline={acc_trivial:.3f}): "
            "esto es indicio de data leakage, no de buen desempeño."
        )


def verificar_acf_valida(ns: dict) -> None:
    """Valida la implementación manual de autocorrelación de la sección de EDA (R2)."""
    for nombre in ("acf_ret", "acf_abs"):
        acf = ns[nombre]
        assert acf[0] == 1.0, f"{nombre}[0] debe ser 1.0 (autocorrelación a rezago 0), da {acf[0]}"
        assert np.all(acf <= 1.0 + 1e-9) and np.all(acf >= -1.0 - 1e-9), (
            f"{nombre} tiene valores fuera de [-1, 1]: la autocorrelación no puede excederlos"
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--notebook", type=Path, default=DEFAULT_NOTEBOOK,
        help=f"Notebook a verificar (default: {DEFAULT_NOTEBOOK.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--dias", type=int, default=250,
        help="Días hábiles de datos sintéticos a generar (default: 250)",
    )
    args = parser.parse_args()

    if not args.notebook.exists():
        sys.exit(f"No existe {args.notebook}. Corré 'make build' primero.")

    print("[1/6] test_voz: buscando voz de asistente en celdas markdown no exentas...")
    test_voz()
    print("      OK — sin términos prohibidos (R1) fuera de celdas no-pdf")

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = generar_csv_sintetico(
            Path(tmp) / "SYNTH_minute_data_with_indicators.csv", dias=args.dias,
        )
        print(f"[2/6] CSV sintético generado: {csv_path} ({args.dias} días hábiles, formato Kaggle Nifty)")

        print("[3/6] Ejecutando celdas del notebook (yfinance y Keras mockeados, sin red, sin GPU)...")
        ns = ejecutar_notebook(args.notebook, csv_path)
        print("      OK — todas las celdas de código ejecutaron sin error (incluye EDA, figuras R3 y asserts de alineación)")

        print("[4/6] Verificando purga sin solape entre train y test...")
        verificar_purga_sin_solape(ns)
        print(f"      OK — purga de {ns['PURGA']} muestras, sin solape")

        print("[5/6] Verificando autocorrelación manual de la sección de EDA...")
        verificar_acf_valida(ns)
        print("      OK — acf_ret[0] = acf_abs[0] = 1.0, valores en [-1, 1]")

        print("[6/6] Sanity check anti-leakage (datos sintéticos sin señal real)...")
        verificar_sanity_anti_leakage(ns)
        print(
            f"      OK — LogReg={ns['acc_lr']:.3f}  MLP={ns['acc_mlp']:.3f}  "
            f"baseline={ns['acc_trivial']:.3f}"
        )

    print("\nverify_pipeline: TODO OK")


if __name__ == "__main__":
    main()
