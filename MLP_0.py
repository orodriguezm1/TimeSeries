# ================================================================
# Dataset: CO2 mensual (statsmodels)
# ================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# -----------------------------
# 1) Cargar serie CO2 mensual
# -----------------------------
co2 = sm.datasets.co2.load_pandas().data.rename(columns={"co2": "y"})
co2.index = pd.to_datetime(co2.index)
df = co2["y"].resample("ME").mean().interpolate("linear").dropna().to_frame()

# -----------------------------
# 2) Crear features SIN REZAGOS
#    - t_norm: índice temporal normalizado [0,1]
#    - mes_cíclico (opcional): sin/cos(2π * mes/12)
# -----------------------------
df = df.copy()
t = np.arange(len(df))
df["t_norm"] = (t - t.min()) / (t.max() - t.min())  # tiempo normalizado

# estacionalidad mensual (sin lags)
month = df.index.month.values
df["m_sin"] = np.sin(2 * np.pi * month / 12.0)
df["m_cos"] = np.cos(2 * np.pi * month / 12.0)

# -----------------------------
# 3) Split train/test por tiempo
# -----------------------------
split_idx = int(len(df) * 0.85)
train = df.iloc[:split_idx].copy()
test  = df.iloc[split_idx:].copy()

# -----------------------------
# 4) Dos conjuntos de features (ambos SIN ventanas)
# -----------------------------
X_train_time = train[["t_norm"]].values
X_test_time  = test[["t_norm"]].values

X_train_time_season = train[["t_norm", "m_sin", "m_cos"]].values
X_test_time_season  = test[["t_norm", "m_sin", "m_cos"]].values

y_train = train["y"].values
y_test  = test["y"].values

# Escalamos y para estabilizar entrenamiento
ysc = StandardScaler()
y_train_sc = ysc.fit_transform(y_train.reshape(-1,1)).ravel()
y_test_sc  = ysc.transform(y_test.reshape(-1,1)).ravel()

# -----------------------------
# 5) Definir y entrenar MLPs (SIN ventanas)
# -----------------------------
mlp_time = MLPRegressor(
    hidden_layer_sizes=(64,64),
    activation='relu',
    solver='adam',
    learning_rate='adaptive',
    max_iter=2000,
    early_stopping=True,
    n_iter_no_change=20,
    random_state=42
)
mlp_time.fit(X_train_time, y_train_sc)

mlp_time_season = MLPRegressor(
    hidden_layer_sizes=(64,64),
    activation='relu',
    solver='adam',
    learning_rate='adaptive',
    max_iter=2000,
    early_stopping=True,
    n_iter_no_change=20,
    random_state=42
)
mlp_time_season.fit(X_train_time_season, y_train_sc)

# -----------------------------
# 6) Predicciones y métricas
# -----------------------------
pred_time_sc = mlp_time.predict(X_test_time)
pred_seas_sc = mlp_time_season.predict(X_test_time_season)

# des-escalar a la magnitud original
pred_time = ysc.inverse_transform(pred_time_sc.reshape(-1,1)).ravel()
pred_seas = ysc.inverse_transform(pred_seas_sc.reshape(-1,1)).ravel()

rmse_time = np.sqrt(mean_squared_error(y_test, pred_time))
rmse_seas = np.sqrt(mean_squared_error(y_test, pred_seas))

print(f"RMSE SIN ventanas — Solo tiempo:          {rmse_time:.3f}")
print(f"RMSE SIN ventanas — Tiempo + estacional.: {rmse_seas:.3f}")

# -----------------------------
# 7) Gráficos comparativos
# -----------------------------
plt.figure(figsize=(12,6))
plt.plot(train.index, y_train, label="Train (real)")
plt.plot(test.index,  y_test,  label="Test (real)")

plt.plot(test.index, pred_time, linestyle="--", label=f"MLP sin ventanas (solo tiempo) | RMSE={rmse_time:.2f}")
plt.plot(test.index, pred_seas, linestyle="--", label=f"MLP sin ventanas (tiempo+estac.) | RMSE={rmse_seas:.2f}")

plt.title("Pronóstico SIN ventanas deslizantes (MLP)\nCO₂ mensual")
plt.xlabel("Tiempo"); plt.ylabel("ppm CO₂")
plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
plt.show()

# -----------------------------
# 8) Comentario:
#    - Estos modelos NO usan rezagos, así que no 'ven' la dinámica auto-regresiva.
#    - El de 'tiempo+estacionalidad' puede aproximar tendencia y ciclo anual,
#      pero sin memoria explícita suele rendir peor que un MLP/LSTM con ventanas.
# -----------------------------
