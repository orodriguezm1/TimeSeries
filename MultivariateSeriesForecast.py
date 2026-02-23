import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

# ---------------------------------------------------------------------------
# Inicialización
# ---------------------------------------------------------------------------
tf.random.set_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Carga y limpieza de datos
# ---------------------------------------------------------------------------
df = pd.read_csv("indicadores_causales.csv", parse_dates=["Fecha"])
df = df.sort_values("Fecha").reset_index(drop=True)

print("Columnas disponibles:", list(df.columns))
if "USD_COP" not in df.columns:
    raise ValueError("La columna 'USD_COP' no está presente en el archivo CSV.")

# Limpieza numérica
for c in df.columns:
    if c != "Fecha":
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.interpolate(method="linear").bfill().ffill()


# ---------------------------------------------------------------------------
# Parámetros
# ---------------------------------------------------------------------------
LOOKBACK = 15
TRAIN_RATIO = 11 / 12  # aprox. 11 meses train / 1 mes test

TARGET_COL_NAME = "USD_COP"
FEATURE_COLS = [c for c in df.columns if c != "Fecha"]
target_idx = FEATURE_COLS.index(TARGET_COL_NAME)

# ---------------------------------------------------------------------------
# Split temporal ANTES de normalizar (evita leakage)
# ---------------------------------------------------------------------------
split_raw = int(len(df) * TRAIN_RATIO)  # índice en df donde empieza el test
df_train = df.iloc[:split_raw].copy()
df_test = df.iloc[split_raw:].copy()

train_data = df_train[FEATURE_COLS].values.astype(np.float32)
test_data = df_test[FEATURE_COLS].values.astype(np.float32)

# ---------------------------------------------------------------------------
# Normalización SOLO con train (evita leakage)
# ---------------------------------------------------------------------------
mean_val = train_data.mean(axis=0)
std_val = train_data.std(axis=0)
std_val[std_val == 0] = 1.0

train_norm = (train_data - mean_val) / std_val
test_norm = (test_data - mean_val) / std_val

n_features = train_norm.shape[1]

# ---------------------------------------------------------------------------
# Dataset secuencial
# ---------------------------------------------------------------------------
def make_sequences(data_norm, lookback, y_index):
    X, y = [], []
    for i in range(len(data_norm) - lookback):
        X.append(data_norm[i:i + lookback])
        y.append(data_norm[i + lookback, y_index])
    return np.array(X, np.float32), np.array(y, np.float32).reshape(-1, 1)

# Entrenamiento: secuencias solo del TRAIN
X_train_full, y_train_full = make_sequences(train_norm, LOOKBACK, target_idx)

# Validación: tomar un pedazo del train (NO usar test como validation)
val_ratio = 0.1
val_cut = int(len(X_train_full) * (1 - val_ratio))
X_tr, X_val = X_train_full[:val_cut], X_train_full[val_cut:]
y_tr, y_val = y_train_full[:val_cut], y_train_full[val_cut:]

print(f"X_train_full shape: {X_train_full.shape} | y_train_full shape: {y_train_full.shape}")
print(f"Train: {df_train['Fecha'].iloc[0].date()} → {df_train['Fecha'].iloc[-1].date()}")
print(f"Test:  {df_test['Fecha'].iloc[0].date()} → {df_test['Fecha'].iloc[-1].date()}")

# ---------------------------------------------------------------------------
# Definición del modelo GRU multivariado
# ---------------------------------------------------------------------------
base = tf.keras.Sequential(name="GRU_multivariada")
base.add(tf.keras.Input(shape=(LOOKBACK, n_features), name="Input"))

base.add(tf.keras.layers.GRU(
    units=64,
    activation="tanh",
    return_sequences=True,
    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
    name="GRU_1"
))
base.add(tf.keras.layers.GRU(
    units=8,
    activation="tanh",
    return_sequences=False,
    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
    name="GRU_2"
))
base.add(tf.keras.layers.Dense(1, activation="linear", name="Output"))


opt = Adam(learning_rate=1e-4)
base.compile(optimizer=opt, loss="mse")

# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
print("\nEntrenando modelo...\n")
history = base.fit(
    X_tr, y_tr,
    batch_size=50,
    epochs=1000,
    verbose=1,
    validation_data=(X_val, y_val),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=25,
            restore_best_weights=True
        )
    ]
)

# ---------------------------------------------------------------------------
# Predicción CONTINUA (como tu gráfico), pero:
# - En TRAIN: predicción 1-step (ajuste)
# - En TEST: forecast REAL recursivo (sin usar USD/COP real del test en la ventana)
# ---------------------------------------------------------------------------

# 1) Predicción 1-step en el tramo TRAIN (ajuste)
preds_train_norm = base.predict(X_train_full, verbose=0)  # (len(train)-LOOKBACK, 1)

# 2) Forecast REAL recursivo para el TEST
window = train_norm[-LOOKBACK:].copy()  # (LOOKBACK, n_features) últimos LOOKBACK reales del train
preds_test_norm = []

for t in range(len(test_norm)):
    x_in = window.reshape(1, LOOKBACK, n_features)
    yhat = base.predict(x_in, verbose=0)[0, 0]
    preds_test_norm.append(yhat)

    # Exógenas reales del test, pero USD/COP (target) se reemplaza por la predicción
    next_row = test_norm[t].copy()
    next_row[target_idx] = yhat

    # avanzar ventana
    window = np.vstack([window[1:], next_row])

preds_test_norm = np.array(preds_test_norm, dtype=np.float32).reshape(-1, 1)

# 3) Unir para que NO haya "disconexión"
preds_all_norm = np.vstack([preds_train_norm, preds_test_norm])

# Fechas y real completo (desde LOOKBACK)
dates_all = df["Fecha"].iloc[LOOKBACK:].reset_index(drop=True)
Y_all_real = df[TARGET_COL_NAME].values.astype(np.float32)[LOOKBACK:].reshape(-1, 1)

# Sanity check: misma longitud
assert len(preds_all_norm) == len(dates_all) == len(Y_all_real)

# Desnormalizar predicción (solo target)
preds_all_denorm = preds_all_norm * std_val[target_idx] + mean_val[target_idx]
Y_all_denorm = Y_all_real  # ya está en escala real

# Split date exacta (primera fecha del test)
split_date = df["Fecha"].iloc[split_raw]

# ---------------------------------------------------------------------------
# Gráfica continua: Real completa + Predicho continuo + línea vertical
# ---------------------------------------------------------------------------
plt.figure(figsize=(12, 6))

plt.plot(dates_all, Y_all_denorm, 'b-', label="Real USD/COP")
plt.plot(dates_all, preds_all_denorm, 'r--', label="Predicho (ajuste + forecast real)")

plt.axvline(split_date, color='gray', linestyle='--', linewidth=1)
plt.text(
    split_date, plt.ylim()[1] * 0.99, "← Entrenamiento | Predicción →",
    ha='center', va='top', color='gray', fontsize=10
)

plt.title("Ajuste (train) y forecast real (test) del USD/COP usando GRU multivariada")
plt.xlabel("Fecha")
plt.ylabel("USD/COP")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# Gráfica de pérdida
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 4))
plt.semilogy(history.history['loss'], label="Train Loss (MSE)")
plt.semilogy(history.history['val_loss'], label="Validation Loss")
plt.xlabel("Época")
plt.ylabel("Pérdida (log)")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# Métricas del TEST (solo para evaluar)
# ---------------------------------------------------------------------------
# Predicciones del test en escala real:
preds_test_denorm = preds_test_norm * std_val[target_idx] + mean_val[target_idx]
y_test_real = test_data[:, target_idx].reshape(-1, 1)

mae = np.mean(np.abs(preds_test_denorm - y_test_real))
rmse = np.sqrt(np.mean((preds_test_denorm - y_test_real) ** 2))
print(f"\nMAE test (forecast real):  {mae:.4f}")
print(f"RMSE test (forecast real): {rmse:.4f}")
