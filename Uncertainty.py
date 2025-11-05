# ===========================================================================
# Predicción de series de tiempo con incertidumbre:
#   RNN, GRU y LSTM (Monte Carlo Dropout)
# ===========================================================================
import numpy as np
import pandas as pd
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Oculta mensajes INFO y WARNING de TensorFlow
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, Model
from time import time

# ---------------------------------------------------------------------------
# Inicialización reproducible
# ---------------------------------------------------------------------------
tf.random.set_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Carga y limpieza de datos
# ---------------------------------------------------------------------------
df = pd.read_csv("indicadores_causales.csv", parse_dates=["Fecha"])
df = df.sort_values("Fecha").reset_index(drop=True)

if "USD_COP" not in df.columns:
    raise ValueError("La columna 'USD_COP' no está presente en el archivo CSV.")

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
# Normalización y dataset secuencial
# ---------------------------------------------------------------------------
data = df.drop(columns=["Fecha"]).values.astype(np.float32)
mean_val = np.mean(data, axis=0)
std_val = np.std(data, axis=0)
data_norm = (data - mean_val) / std_val

LOOKBACK = 15
X, Y = [], []
for i in range(len(data_norm) - LOOKBACK):
    X.append(data_norm[i:i + LOOKBACK])
    Y.append(data_norm[i + LOOKBACK, 0])
X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32).reshape(-1, 1)
dates = df["Fecha"].iloc[LOOKBACK:].reset_index(drop=True)

split_idx = int(len(dates) * (11 / 12))
X_train, X_test = X[:split_idx], X[split_idx:]
Y_train, Y_test = Y[:split_idx], Y[split_idx:]
dates_train, dates_test = dates[:split_idx], dates[split_idx:]
n_features = X.shape[2]

# ---------------------------------------------------------------------------
# Función para crear modelos con Dropout (para incertidumbre)
# ---------------------------------------------------------------------------
def build_model(cell_type="RNN"):
    inputs = tf.keras.Input(shape=(LOOKBACK, n_features))
    x = None

    if cell_type == "RNN":
        x = layers.SimpleRNN(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs)
        x = layers.SimpleRNN(16, dropout=0.2)(x)
    elif cell_type == "LSTM":
        x = layers.LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs)
        x = layers.LSTM(16, dropout=0.2)(x)
    elif cell_type == "GRU":
        x = layers.GRU(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs)
        x = layers.GRU(16, dropout=0.2)(x)

    outputs = layers.Dense(1, activation="linear")(x)
    model = tf.keras.Model(inputs, outputs, name=f"{cell_type}_MC_Dropout")
    return model


# ---------------------------------------------------------------------------
# Entrenamiento de cada modelo
# ---------------------------------------------------------------------------
def train_model(model, X_train, Y_train, X_val, Y_val, epochs=500):
    model.compile(optimizer=Adam(1e-4), loss="mse")
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                        epochs=epochs, batch_size=50, verbose=0)
    return history

models = {}
histories = {}

for cell in ["RNN", "GRU", "LSTM"]:
    print(f"\nEntrenando modelo {cell}...")
    model = build_model(cell)
    hist = train_model(model, X_train, Y_train, X_test, Y_test, epochs=500)
    models[cell] = model
    histories[cell] = hist

# ---------------------------------------------------------------------------
# Monte Carlo Dropout: muestreo de incertidumbre
# ---------------------------------------------------------------------------
def mc_dropout_predictions(model, X, n_samples=100):
    preds = [model(X, training=True).numpy().flatten() for _ in range(n_samples)]
    preds = np.array(preds)
    return preds.mean(axis=0), preds.std(axis=0)

# ---------------------------------------------------------------------------
# Predicciones y desnormalización
# ---------------------------------------------------------------------------
predictions = {}
for cell in ["RNN", "GRU", "LSTM"]:
    print(f"Calculando incertidumbre ({cell})...")
    mean_pred, std_pred = mc_dropout_predictions(models[cell], X)
    mean_denorm = mean_pred * std_val[0] + mean_val[0]
    std_denorm = std_pred * std_val[0]
    true_denorm = Y.flatten() * std_val[0] + mean_val[0]
    predictions[cell] = (mean_denorm, std_denorm, true_denorm)

# ---------------------------------------------------------------------------
# Gráficos: predicción con incertidumbre
# ---------------------------------------------------------------------------
for cell in ["RNN", "GRU", "LSTM"]:
    mean_denorm, std_denorm, true_denorm = predictions[cell]

    plt.figure(figsize=(12, 5))
    plt.plot(dates, true_denorm, color="blue", label="Real USD/COP", linewidth=1.5)
    plt.plot(dates, mean_denorm, color="red", linestyle="--", label=f"Predicción ({cell})")
    plt.fill_between(dates,
                     mean_denorm - 1.96 * std_denorm,
                     mean_denorm + 1.96 * std_denorm,
                     color="red", alpha=0.15,
                     label="Intervalo 95% (MC Dropout)")

    split_date = dates.iloc[split_idx]
    plt.axvline(split_date, color="gray", linestyle="--", linewidth=1)
    plt.text(split_date, plt.ylim()[1]*0.98,
             "← Entrenamiento | Predicción →",
             ha="center", va="top", color="gray", fontsize=10)

    plt.title(f"Predicción del USD/COP con incertidumbre ({cell})")
    plt.xlabel("Fecha")
    plt.ylabel("USD/COP")
    plt.legend()
    plt.tight_layout()
    plt.show()
