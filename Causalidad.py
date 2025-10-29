"""
==============================================================
Causalidad comparativa en series de tiempo
==============================================================
Descripción:
Este script compara tres métodos de detección de causalidad
entre series económicas:
    1. Causalidad de Granger (lineal)
    2. Entropía de Transferencia (no lineal, teoría de la información)
    3. Causalidad de Granger no lineal (MLP y LSTM)

Dependencias:
pip install yfinance statsmodels pyinform scikit-learn tensorflow
==============================================================
"""

# ============================================================
# LIBRERÍAS
# ============================================================
import warnings
warnings.filterwarnings("ignore", message=".*verbose is deprecated.*")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta

# Granger lineal
from statsmodels.tsa.stattools import grangercausalitytests

# Entropía de transferencia (discreta)
import pyinform.transferentropy as te

# Redes neuronales
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler

# ============================================================
# DESCARGA Y PREPROCESAMIENTO DE DATOS
# ============================================================

def descargar_cierre(tickers, start, end):
    """Descarga precios de cierre ajustados de Yahoo Finance."""
    datos = {}
    for nombre, ticker in tickers.items():
        df = yf.download(ticker, start=start, end=end, interval="1d",
                         auto_adjust=True, progress=False)
        if df.empty:
            raise RuntimeError(f"No se obtuvieron datos para {ticker}")
        col_cierre = [c for c in df.columns if "Close" in c][0]
        datos[nombre] = df[col_cierre].copy()
    return pd.DataFrame(datos).dropna(how="all")

# Rango temporal
end = date.today()
start = end - timedelta(days=365)
print(f"Descargando datos desde {start} hasta {end}...\n")

tickers = {
    "USD_COP": "USDCOP=X",
    "Brent_USD": "BZ=F",
    "FED_RATE": "^IRX",
    "VIX": "^VIX",
    "SP500": "^GSPC"
}

df = descargar_cierre(tickers, start, end)
df["Brent_COP"] = df["USD_COP"] * df["Brent_USD"]
df = df[["USD_COP", "Brent_COP", "FED_RATE", "VIX", "SP500"]].dropna()
df.index.name = "Fecha"

print("Datos descargados (últimas filas):\n", df.tail(), "\n")

# ============================================================
# TRANSFORMACIONES
# ============================================================
data_diff = np.log(df).diff().dropna()

pares = [
    ("USD_COP", "Brent_COP"),
    ("USD_COP", "FED_RATE"),
    ("USD_COP", "VIX"),
    ("USD_COP", "SP500")
]

# ============================================================
# 1) GRANGER LINEAL
# ============================================================
def test_granger(df, y_var, x_var, maxlag=5):
    resultado = grangercausalitytests(df[[y_var, x_var]], maxlag=maxlag, verbose=False)
    pvals = [resultado[i+1][0]['ssr_ftest'][1] for i in range(maxlag)]
    return min(pvals)

granger_results = {}
print("=== Causalidad de Granger (lineal) ===")
for dep, indep in pares:
    try:
        p = test_granger(data_diff, dep, indep)
        granger_results[(indep, dep)] = p
        print(f"{indep} → {dep}: p = {p:.4f} {'**Causal**' if p < 0.05 else ''}")
    except Exception as e:
        granger_results[(indep, dep)] = np.nan
        print(f"{indep} → {dep}: error ({e})")
print()

# ============================================================
# 2) ENTROPÍA DE TRANSFERENCIA (pyinform)
# ============================================================
def discretizar_por_cuantiles(x, bins=5):
    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(x, qs))
    if len(edges) - 1 < bins:
        edges = np.histogram_bin_edges(x, bins=bins)
    return np.digitize(x, edges[1:-1], right=False).astype(int)

def transfer_entropy_pyinform(x, y, k=1, bins=5):
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    n = min(len(x), len(y))
    xd, yd = discretizar_por_cuantiles(x[:n], bins=bins), discretizar_por_cuantiles(y[:n], bins=bins)
    try:
        val = te.transferentropy(xd, yd, k=k)
    except Exception:
        val = 0.0
    return float(val)

te_results = {}
print("=== Entropía de Transferencia (no lineal) ===")
for dep, indep in pares:
    x, y = data_diff[indep].dropna().values, data_diff[dep].dropna().values
    te_xy = transfer_entropy_pyinform(x, y, k=2, bins=8)
    te_yx = transfer_entropy_pyinform(y, x, k=2, bins=8)
    te_results[(indep, dep)] = te_xy
    print(f"{indep} → {dep}: TE = {te_xy:.4f} bits | {dep} → {indep}: TE = {te_yx:.4f} bits")
print()

# ============================================================
# 3) GRANGER NO LINEAL CON REDES (MLP y LSTM)
# ============================================================
def construir_matriz_lags(y, x=None, p=5):
    y, T = np.asarray(y).ravel(), len(y)
    if x is not None:
        x = np.asarray(x).ravel()
        T = min(T, len(x))
        y, x = y[-T:], x[-T:]
    Ylags = np.column_stack([np.roll(y, i) for i in range(1, p+1)])
    mask = np.ones(T, dtype=bool)
    mask[:p] = False
    Xr, y_target = Ylags[mask], y[mask]
    if x is not None:
        Xlags = np.column_stack([np.roll(x, i) for i in range(1, p+1)])[mask]
        Xf = np.concatenate([Xr, Xlags], axis=1)
    else:
        Xf = Xr.copy()
    return Xr, Xf, y_target

def build_mlp(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_lstm(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(32),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def mlp_granger(y, x, p=5, test_size=0.2, epochs=150):
    Xr, Xf, y_t = construir_matriz_lags(y, x, p=p)
    y_s = StandardScaler().fit_transform(y_t.reshape(-1, 1))
    Xr_s, Xf_s = StandardScaler().fit_transform(Xr), StandardScaler().fit_transform(Xf)
    n, cut = len(y_s), int(len(y_s)*(1-test_size))
    Xr_tr, Xr_te, yr_tr, yr_te = Xr_s[:cut], Xr_s[cut:], y_s[:cut], y_s[cut:]
    Xf_tr, Xf_te = Xf_s[:cut], Xf_s[cut:]
    mlp_r, mlp_f = build_mlp(Xr_tr.shape[1]), build_mlp(Xf_tr.shape[1])
    mlp_r.fit(Xr_tr, yr_tr, epochs=epochs, verbose=0)
    mlp_f.fit(Xf_tr, yr_tr, epochs=epochs, verbose=0)
    mse_r, mse_f = mlp_r.evaluate(Xr_te, yr_te, verbose=0), mlp_f.evaluate(Xf_te, yr_te, verbose=0)
    return (mse_r - mse_f) / (mse_r + 1e-12) * 100.0

def lstm_granger(y, x, p=5, test_size=0.2, epochs=80):
    Xr, Xf, y_t = construir_matriz_lags(y, x, p=p)
    y_s = StandardScaler().fit_transform(y_t.reshape(-1, 1))
    Xr_s, Xf_s = StandardScaler().fit_transform(Xr), StandardScaler().fit_transform(Xf)
    N, cut = len(y_s), int(len(y_s)*(1-test_size))
    Xr_seq, Xf_seq = Xr_s.reshape(N, p, 1), np.stack([Xf_s[:, :p], Xf_s[:, p:]], axis=2)
    Xr_tr, Xr_te, yr_tr, yr_te = Xr_seq[:cut], Xr_seq[cut:], y_s[:cut], y_s[cut:]
    Xf_tr, Xf_te = Xf_seq[:cut], Xf_seq[cut:]
    lstm_r, lstm_f = build_lstm((p, 1)), build_lstm((p, 2))
    lstm_r.fit(Xr_tr, yr_tr, epochs=epochs, verbose=0)
    lstm_f.fit(Xf_tr, yr_tr, epochs=epochs, verbose=0)
    mse_r, mse_f = lstm_r.evaluate(Xr_te, yr_te, verbose=0), lstm_f.evaluate(Xf_te, yr_te, verbose=0)
    return (mse_r - mse_f) / (mse_r + 1e-12) * 100.0

mlp_results, lstm_results = {}, {}
print("=== Causalidad de Granger no lineal (MLP y LSTM) ===")
for dep, indep in pares:
    y, x = data_diff[dep].values, data_diff[indep].values
    n = min(len(y), len(x))
    try:
        imp_mlp = mlp_granger(y[-n:], x[-n:])
        imp_lstm = lstm_granger(y[-n:], x[-n:])
        mlp_results[(indep, dep)] = imp_mlp
        lstm_results[(indep, dep)] = imp_lstm
        print(f"{indep} → {dep}: Mejora MLP = {imp_mlp:6.2f}% | Mejora LSTM = {imp_lstm:6.2f}%")
    except Exception as e:
        mlp_results[(indep, dep)] = np.nan
        lstm_results[(indep, dep)] = np.nan
        print(f"{indep} → {dep}: error ({e})")
print()

# ============================================================
# 4) RESUMEN FINAL
# ============================================================
print("=====================================================")
print("Resumen comparativo de causalidades detectadas:\n")
summary = []
for dep, indep in pares:
    g_p = granger_results.get((indep, dep), np.nan)
    te_v = te_results.get((indep, dep), np.nan)
    m_mlp = mlp_results.get((indep, dep), np.nan)
    m_lstm = lstm_results.get((indep, dep), np.nan)
    summary.append([f"{indep}→{dep}", g_p, te_v, m_mlp, m_lstm])

df_summary = pd.DataFrame(summary, columns=["Relación", "p-Granger", "TE (bits)", "Mejora MLP (%)", "Mejora LSTM (%)"])
print(df_summary.to_string(index=False, float_format="%.4f"))
print("=====================================================\n")

print("Interpretación:")
print("- Granger lineal: p < 0.05 → causalidad significativa.")
print("- Entropía de Transferencia: TE > 0.05 bits → dependencia informacional.")
print("- Granger no lineal (MLP/LSTM): mejora > 5–10% → posible causalidad no lineal.\n")
