import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from time import time

# ---------------------------------------------------------------------------
# Inicialización
# ---------------------------------------------------------------------------
start_time = time()
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

# --- LIMPIEZA DE DATOS ---
for c in df.columns:
    if c != "Fecha":
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.interpolate(method="linear").fillna(method="bfill").fillna(method="ffill")

# ---------------------------------------------------------------------------
# Normalización
# ---------------------------------------------------------------------------
data = df.drop(columns=["Fecha"]).values.astype(np.float32)
n_features = data.shape[1]
mean_val = np.mean(data, axis=0)
std_val = np.std(data, axis=0)
data_norm = (data - mean_val) / std_val

# ---------------------------------------------------------------------------
# Dataset secuencial (lookback)
# ---------------------------------------------------------------------------
LOOKBACK = 15

X, Y = [], []
for i in range(len(data_norm) - LOOKBACK):
    X.append(data_norm[i:i + LOOKBACK])
    Y.append(data_norm[i + LOOKBACK, 0])
X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32).reshape(-1, 1)

dates = df["Fecha"].iloc[LOOKBACK:].reset_index(drop=True)
print(f"X shape: {X.shape} | Y shape: {Y.shape}")

# ---------------------------------------------------------------------------
# División: 11 meses entrenamiento / 1 mes prueba
# ---------------------------------------------------------------------------
# Detectar rango temporal
total_days = len(dates)
split_idx = int(total_days * (11 / 12))  # ≈ 91.6% para entrenar
print(f"\nEntrenamiento: {dates.iloc[0].date()} → {dates.iloc[split_idx-1].date()}")
print(f"Prueba: {dates.iloc[split_idx].date()} → {dates.iloc[-1].date()}")

X_train, X_test = X[:split_idx], X[split_idx:]
Y_train, Y_test = Y[:split_idx], Y[split_idx:]
dates_train, dates_test = dates[:split_idx], dates[split_idx:]

# ---------------------------------------------------------------------------
# Definición del modelo GRU multivariado
# ---------------------------------------------------------------------------
model_ = tf.keras.Sequential(name="GRU_multivariada")

model_.add(tf.keras.layers.GRU(
    units=32,
    activation="tanh",
    return_sequences=True,
    input_shape=(LOOKBACK, n_features),
    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
    name="GRU_1"
))
model_.add(tf.keras.layers.GRU(
    units=16,
    activation="tanh",
    return_sequences=False,
    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
    name="GRU_2"
))
model_.add(tf.keras.layers.Dense(1, activation="linear", name="Output"))

# ---------------------------------------------------------------------------
# Modelo wrapper con pérdida personalizada (MSE)
# ---------------------------------------------------------------------------
class MyModel(tf.keras.Model):
    def __init__(self, base_model):
        super(MyModel, self).__init__()
        self.loc_net = base_model

    def call(self, x):
        return self.loc_net(x)

    @staticmethod
    def MyLoss(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    @staticmethod
    def MyMet(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true / (y_true + 0.1) - y_pred / (y_true + 0.1)))

model = MyModel(model_)

opt = Adam(learning_rate=1e-4, epsilon=1e-16)
model.compile(optimizer=opt,
              loss=model.MyLoss,
              metrics=[model.MyLoss, model.MyMet])

# ---------------------------------------------------------------------------
# Entrenamiento (solo 11 meses)
# ---------------------------------------------------------------------------
print("\nEntrenando modelo...\n")
history = model.fit(
    X_train, Y_train,
    batch_size=50,
    epochs=1000,
    verbose=1,
    validation_data=(X_test, Y_test)
)

# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------
preds_all = model(X).numpy()

# Desnormalización (solo para USD/COP)
preds_all_denorm = preds_all * std_val[0] + mean_val[0]
Y_all_denorm = Y * std_val[0] + mean_val[0]

# Fechas correspondientes
dates_all = df["Fecha"].iloc[LOOKBACK:].reset_index(drop=True)

# ---------------------------------------------------------------------------
# Gráfica continua: ajuste (entrenamiento) + predicción (último mes)
# ---------------------------------------------------------------------------
plt.figure(figsize=(12, 6))

# Serie real completa
plt.plot(dates_all, Y_all_denorm, 'b-', label="Real USD/COP")

# Predicción del modelo en toda la serie
plt.plot(dates_all, preds_all_denorm, 'r--', label="Predicho (modelo)")

# Línea vertical separando entrenamiento y predicción
split_date = dates_all.iloc[split_idx]
plt.axvline(split_date, color='gray', linestyle='--', linewidth=1)
plt.text(split_date, plt.ylim()[1]*0.99, "← Entrenamiento | Predicción →",
         ha='center', va='top', color='gray', fontsize=10)

# Detalles
plt.title("Ajuste (11 meses) y predicción (último mes) del USD/COP usando GRU multivariada")
plt.xlabel("Fecha")
plt.ylabel("USD/COP")
plt.legend()
plt.tight_layout()
plt.show()

# Pérdida
plt.figure(figsize=(8, 4))
plt.semilogy(history.history['loss'], label="Train Loss (MSE)")
plt.semilogy(history.history['val_loss'], label="Validation Loss")
plt.xlabel("Época")
plt.ylabel("Pérdida (log)")
plt.legend()
plt.tight_layout()
plt.show()


