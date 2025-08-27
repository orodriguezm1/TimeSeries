# ================================================================
# Pronóstico de INFLACIÓN trimestral (dataset macro multivariado)
# Comparativa:
#   (1) MLP SIN ventanas (X_{t-1} -> y_t)
#   (2) MLP CON ventanas multivariadas (look_back=8)
#   (3) LSTM multivariada (look_back=8)
# Dataset: statsmodels.datasets.macrodata (USA trimestral)
# ================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

import statsmodels.api as sm
import tensorflow as tf

# -----------------------------
# Utilidades
# -----------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred, eps=1e-8):
    return mean_absolute_percentage_error(y_true + eps, y_pred + eps)

def make_seq_3d(X_2d, y_1d, look_back):
    """
    Convierte X_2d (N, F) e y_1d (N,) en secuencias (N-look_back, look_back, F)
    y targets alineados. X_seq[i] = X_2d[i-look_back:i, :], y_seq[i] = y_1d[i]
    """
    Xs, ys = [], []
    for t in range(look_back, len(X_2d)):
        Xs.append(X_2d[t-look_back:t, :])
        ys.append(y_1d[t])
    return np.array(Xs), np.array(ys)

# -----------------------------
# 1) Cargar dataset macro trimestral
# -----------------------------
data = sm.datasets.macrodata.load_pandas().data.copy()

# Columnas: year, quarter, realgdp, realcons, realinv, realgovt,
# realdpi, cpi, m1, tbilrate, unemp, pop, infl, realint
# Crear índice temporal trimestral
periods = pd.period_range(start=f"{int(data['year'].iloc[0])}Q{int(data['quarter'].iloc[0])}",
                          periods=len(data), freq='Q')
df = data.set_index(periods)
df.index = df.index.to_timestamp()  # índice datetime trimestral
df = df.drop(columns=['year', 'quarter'])

target = "infl"
feature_cols = ['realgdp','realcons','realinv','realdpi','cpi','m1','tbilrate','unemp','pop','realint']

# -----------------------------
# 2) Split temporal global (85%/15%)
# -----------------------------
split_idx = int(len(df) * 0.85)
train = df.iloc[:split_idx].copy()
test  = df.iloc[split_idx:].copy()

# ================================================================
# (1) MLP SIN ventanas: usar X_{t-1} -> y_t (evita fuga de info)
# ================================================================
X_all = df[feature_cols].shift(1)   # lag 1 en todas las features
y_all = df[target].copy()           # y_t

mask_valid = ~X_all.isna().any(axis=1)
X_all = X_all.loc[mask_valid]
y_all = y_all.loc[mask_valid]

# Partición temporal (sobre X_all/y_all ya laggeados)
split_ts_no_window = X_all.index[int(len(X_all) * 0.85)]
X_tr_1 = X_all.loc[:split_ts_no_window].values
y_tr_1 = y_all.loc[:split_ts_no_window].values
X_te_1 = X_all.loc[split_ts_no_window:].values
y_te_1 = y_all.loc[split_ts_no_window:].values
idx_no_window = y_all.loc[split_ts_no_window:].index  # para gráficas

# Escalado (fit en train)
scX_1 = StandardScaler()
scY_1 = StandardScaler()
X_tr_1s = scX_1.fit_transform(X_tr_1)
y_tr_1s = scY_1.fit_transform(y_tr_1.reshape(-1,1)).ravel()
X_te_1s = scX_1.transform(X_te_1)

mlp_nowindow = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    learning_rate='adaptive',
    max_iter=3000,
    early_stopping=True,
    n_iter_no_change=30,
    random_state=42
)
mlp_nowindow.fit(X_tr_1s, y_tr_1s)
pred_1s = mlp_nowindow.predict(X_te_1s)
pred_1  = scY_1.inverse_transform(pred_1s.reshape(-1,1)).ravel()

rmse_1  = rmse(y_te_1, pred_1)
mape_1  = mape(y_te_1, pred_1)

# ================================================================
# (2) MLP CON ventanas multivariadas (look_back=8)
# ================================================================
look_back = 8

X_all2 = df[feature_cols].copy()
y_all2 = df[target].copy()

split_ts = X_all2.index[int(len(X_all2) * 0.85)]
X_tr_2 = X_all2.loc[:split_ts].values
X_te_2 = X_all2.loc[split_ts:].values
y_tr_2 = y_all2.loc[:split_ts].values
y_te_2 = y_all2.loc[split_ts:].values
idx_test_full = X_all2.loc[split_ts:].index  # índice de test para gráficas

# Escalado
scX_2 = StandardScaler()
scY_2 = StandardScaler()
X_tr_2s = scX_2.fit_transform(X_tr_2)
X_te_2s = scX_2.transform(X_te_2)
y_tr_2s = scY_2.fit_transform(y_tr_2.reshape(-1,1)).ravel()
y_te_2s = scY_2.transform(y_te_2.reshape(-1,1)).ravel()

# Secuencias en train
X_seq_tr_2, y_seq_tr_2 = make_seq_3d(X_tr_2s, y_tr_2s, look_back)

# *** CORRECCIÓN CLAVE: usar EXACTAMENTE 'look_back' puntos de contexto ***
# para generar EXACTAMENTE n_test ventanas y alinear 1:1 con y_te_2.
X_concat_te = np.vstack([X_tr_2s[-look_back:], X_te_2s])
y_concat_te = np.hstack([y_tr_2s[-look_back:], y_te_2s])
X_seq_te_2, y_seq_te_2 = make_seq_3d(X_concat_te, y_concat_te, look_back)

# Aplanar para MLP
nF = X_seq_tr_2.shape[-1]
X_seq_tr_2_flat = X_seq_tr_2.reshape(X_seq_tr_2.shape[0], look_back * nF)
X_seq_te_2_flat = X_seq_te_2.reshape(X_seq_te_2.shape[0], look_back * nF)

mlp_window = MLPRegressor(
    hidden_layer_sizes=(256, 128),
    activation='relu',
    solver='adam',
    learning_rate='adaptive',
    max_iter=4000,
    early_stopping=True,
    n_iter_no_change=40,
    random_state=42
)
mlp_window.fit(X_seq_tr_2_flat, y_seq_tr_2)
pred_2s = mlp_window.predict(X_seq_te_2_flat)
pred_2  = scY_2.inverse_transform(pred_2s.reshape(-1,1)).ravel()

# Alineación 1:1 (ya no recortar)
y_te_2_aligned = y_te_2

rmse_2 = rmse(y_te_2_aligned, pred_2)
mape_2 = mape(y_te_2_aligned, pred_2)

# ================================================================
# (3) LSTM multivariada (look_back=8)
# ================================================================
tf.random.set_seed(123)

# Reusar las secuencias 3D
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(look_back, nF)),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
_ = model.fit(
    X_seq_tr_2, y_seq_tr_2,
    epochs=100, batch_size=16, verbose=0,
    validation_split=0.2
)

pred_3s = model.predict(X_seq_te_2, verbose=0).ravel()
pred_3  = scY_2.inverse_transform(pred_3s.reshape(-1,1)).ravel()

rmse_3 = rmse(y_te_2_aligned, pred_3)
mape_3 = mape(y_te_2_aligned, pred_3)

# ================================================================
# Baseline ingenuo: y_t = y_{t-1}
# ================================================================
y_test_full = y_te_2
naive_pred = np.r_[y_tr_2[-1], y_test_full[:-1]]  # último de train, luego shift en test

rmse_naive     = rmse(y_test_full, naive_pred)
mape_naive     = mape(y_test_full, naive_pred)
naive_aligned  = naive_pred  # ya 1:1
rmse_naive_al  = rmse(y_te_2_aligned, naive_aligned)
mape_naive_al  = mape(y_te_2_aligned, naive_aligned)

# ================================================================
# Resultados y gráficas
# ================================================================
print("\n=== Métricas en TEST (infl, trimestral) ===")
print(f"(1) MLP SIN ventanas          -> RMSE={rmse_1:.3f} | MAPE={mape_1*100:.2f}%")
print(f"(2) MLP CON ventanas (L={look_back}) -> RMSE={rmse_2:.3f} | MAPE={mape_2*100:.2f}%")
print(f"(3) LSTM multivariada (L={look_back})-> RMSE={rmse_3:.3f} | MAPE={mape_3*100:.2f}%")
print(f"Naive (test completo)         -> RMSE={rmse_naive:.3f} | MAPE={mape_naive*100:.2f}%")
print(f"Naive (alineado ventanas)     -> RMSE={rmse_naive_al:.3f} | MAPE={mape_naive_al*100:.2f}%")

plt.figure(figsize=(12, 7))
# Curva real (test) para referencia (índice del split global de (2)-(3))
plt.plot(idx_test_full, y_test_full, label="Real (infl)", color="black")

# (1) SIN ventanas (tiene su propio índice por el lag y split interno)
plt.plot(idx_no_window, pred_1, "--", label=f"MLP sin ventanas | RMSE={rmse_1:.2f}")

# (2) CON ventanas (alineado 1:1 con test)
plt.plot(idx_test_full, pred_2, "--", label=f"MLP con ventanas (L={look_back}) | RMSE={rmse_2:.2f}")

# (3) LSTM (alineado 1:1 con test)
plt.plot(idx_test_full, pred_3, "--", label=f"LSTM (L={look_back}) | RMSE={rmse_3:.2f}")

# Baseline naive (alineado con test)
plt.plot(idx_test_full, naive_aligned, ":", label=f"Naive | RMSE={rmse_naive:.2f}", alpha=0.8)

plt.title("Predicción de inflación trimestral (macro multivariado)\nComparativa: sin memoria vs. con memoria")
plt.xlabel("Tiempo"); plt.ylabel("Inflación trimestral (%)")
plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
plt.show()
