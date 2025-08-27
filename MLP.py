# ================================================================
# MLP para series de tiempo con gráficos (univariante)
# Opción B: CO₂ mensual (sin internet, statsmodels)
# ================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings

# -----------------------------
# 1) Cargar serie univariante y preparar índice temporal
# -----------------------------

def load_series_co2():
    """CO2 mensual desde statsmodels (sin internet)."""
    import statsmodels.api as sm
    co2 = sm.datasets.co2.load_pandas().data.rename(columns={"co2": "y"})
    co2.index = pd.to_datetime(co2.index)
    s = co2["y"].resample("ME").mean().interpolate("linear").dropna()
    return s.to_frame()

df = load_series_co2()
source_used = "CO₂ (statsmodels, ppm mensual)"

# -----------------------------
# 2) Split train / test por tiempo (85% / 15%)
# -----------------------------
split_idx = int(len(df) * 0.85)
train = df.iloc[:split_idx].copy()
test  = df.iloc[split_idx:].copy()

# -----------------------------
# 3) Función para construir ventanas deslizantes (supervisado)
#    X[t] = [y_{t-p}, ..., y_{t-1}],   Y[t] = y_t
# -----------------------------
def make_supervised(arr_2d, p):
    """arr_2d: shape (N,1). Devuelve X (N-p, p) y Y (N-p,)."""
    X, Y = [], []
    for t in range(p, len(arr_2d)):
        X.append(arr_2d[t-p:t, 0])
        Y.append(arr_2d[t, 0])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

# -----------------------------
# 4) Escalado SOLO con train (evitar fuga de información)
# -----------------------------
scaler = StandardScaler()
y_tr = scaler.fit_transform(train[["y"]].values)      # (n_train, 1)
y_te = scaler.transform(test[["y"]].values)           # (n_test, 1)
y_full = np.vstack([y_tr, y_te])                      # útil para ventanas de test

# -----------------------------
# 5) Crear ventanas
#    - Para train: sólo dentro de train
#    - Para test: ventanas que comienzan tras el último índice de train,
#                 usando los últimos p puntos de train como contexto
# -----------------------------
p = 12  # tamaño de ventana (p.ej., 12 meses)
X_tr, Y_tr = make_supervised(y_tr, p)

# Índices para construir ventanas de test a partir de y_full
start_test = len(y_tr)  # primer índice en el conjunto completo que pertenece a test
# Construimos ventanas cuyos targets caen en test:
X_te, Y_te = [], []
for t in range(start_test, len(y_full)):
    if t - p < 0:
        continue
    X_te.append(y_full[t-p:t, 0])  # incluye p últimos puntos (posible mezcla: últimos de train + primeros de test)
    Y_te.append(y_full[t, 0])
X_te = np.array(X_te)
Y_te = np.array(Y_te)

# -----------------------------
# 6) MLPRegressor (tu configuración base, con mejoras prácticas)
# -----------------------------
mlp = MLPRegressor(
    hidden_layer_sizes=(50, 50),
    activation='relu',
    solver='adam',
    learning_rate='adaptive',
    max_iter=2000,
    early_stopping=True,
    n_iter_no_change=20,
    random_state=42
)
mlp.fit(X_tr, Y_tr)

# -----------------------------
# 7) Predicciones y des-escalado
# -----------------------------
yhat_tr = mlp.predict(X_tr)           # escala estandarizada
yhat_te = mlp.predict(X_te)

# Invertir escala: necesitamos shape (n,1) para inverse_transform
yhat_tr_inv = scaler.inverse_transform(yhat_tr.reshape(-1, 1)).ravel()
yhat_te_inv = scaler.inverse_transform(yhat_te.reshape(-1, 1)).ravel()

# Alinear verdaderos en escala original
y_tr_inv = train["y"].values[p:]          # por ventana, se pierden los primeros p
y_te_inv = test["y"].values               # en test apuntamos a todos los puntos de test (por construcción de X_te)

# Métricas
rmse_tr = np.sqrt(mean_squared_error(y_tr_inv, yhat_tr_inv))
rmse_te = np.sqrt(mean_squared_error(y_te_inv, yhat_te_inv))
print(f"Fuente: {source_used}")
print(f"RMSE (train): {rmse_tr:.4f}")
print(f"RMSE (test):  {rmse_te:.4f}")

# -----------------------------
# 8) Gráficos
# -----------------------------
plt.figure(figsize=(12, 6))
plt.plot(train.index, train["y"], label="Train (real)")
plt.plot(test.index,  test["y"],  label="Test (real)")

# Predicciones TRAIN: colocar desde el índice p
plt.plot(train.index[p:], yhat_tr_inv, label="MLP (ajuste in-sample)")

# Predicciones TEST: alineadas 1:1 con test.index
plt.plot(test.index, yhat_te_inv, linestyle="--", label="MLP (predicción out-of-sample)")

plt.title(f"MLP para Serie Temporal (ventana p={p})\n{source_used}\nRMSE train={rmse_tr:.3f} | RMSE test={rmse_te:.3f}")
plt.xlabel("Tiempo")
plt.ylabel("Valor")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
