from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib
matplotlib.use("Agg")   # 👈 añade esta línea
import matplotlib.pyplot as plt
import os
from time import time
import json


# ---------------------------------------------------------------------------
# CONFIGURACIÓN GLOBAL
# ---------------------------------------------------------------------------
BASE_PATH = "/Users/orodriguez/PycharmProjects/TimeSeries/"
CONFIG_FILE = os.path.join(BASE_PATH, "config.json")
GRAPH_PATH = os.path.join(BASE_PATH, "grapics")
os.makedirs(GRAPH_PATH, exist_ok=True)

# Cargar configuración desde JSON
if not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError(f"No se encontró el archivo de configuración: {CONFIG_FILE}")

with open(CONFIG_FILE, "r") as f:
    default_config = json.load(f)

app = FastAPI(
    title="Time Series API",
    description="Entrenamiento y predicción multivariada con GRU configurable desde config.json",
    version="1.0.1"
)

# ---------------------------------------------------------------------------
# MODELO DE CONFIGURACIÓN
# ---------------------------------------------------------------------------
class ModelConfig(BaseModel):
    path: str = default_config.get("path")
    target_column: str = default_config.get("target_column")
    architecture: List[int] = default_config.get("architecture")
    epochs: int = default_config.get("epochs", 200)
    days_to_predict: int = default_config.get("days_to_predict", 30)

# ---------------------------------------------------------------------------
# FUNCIONES AUXILIARES
# ---------------------------------------------------------------------------
def load_and_prepare_data(path: str, target_col: str, lookback: int = 15):
    df = pd.read_csv(path, parse_dates=["Fecha"])
    df = df.sort_values("Fecha").reset_index(drop=True)

    if target_col not in df.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no está en el archivo CSV.")

    for c in df.columns:
        if c != "Fecha":
            df[c] = (
                df[c].astype(str)
                .str.replace(",", ".", regex=False)
                .str.strip()
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.interpolate(method="linear").bfill().ffill()

    data = df.drop(columns=["Fecha"]).values.astype(np.float32)
    mean_val = np.mean(data, axis=0)
    std_val = np.std(data, axis=0)
    data_norm = (data - mean_val) / std_val

    n_features = data_norm.shape[1]
    X, Y = [], []
    target_idx = list(df.columns).index(target_col) - 1
    for i in range(len(data_norm) - lookback):
        X.append(data_norm[i:i + lookback])
        Y.append(data_norm[i + lookback, target_idx])

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32).reshape(-1, 1)
    dates = df["Fecha"].iloc[lookback:].reset_index(drop=True)
    return X, Y, dates, mean_val, std_val, n_features, df, target_idx


def build_gru_model(architecture: List[int], lookback: int, n_features: int):
    model = tf.keras.Sequential(name="GRU_multivariada")
    for i, units in enumerate(architecture):
        return_seq = (i < len(architecture) - 1)
        model.add(tf.keras.layers.GRU(
            units=units,
            activation="tanh",
            return_sequences=return_seq,
            input_shape=(lookback, n_features),
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
            name=f"GRU_{i + 1}"
        ))
    model.add(tf.keras.layers.Dense(1, activation="linear", name="Output"))
    return model


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

# ---------------------------------------------------------------------------
# ENDPOINT: ENTRENAMIENTO Y PREDICCIÓN
# ---------------------------------------------------------------------------
@app.post("/train/")
def train_gru(config: ModelConfig = ModelConfig()):
    """
    Entrena un modelo GRU multivariado leyendo automáticamente los parámetros desde config.json
    """
    start_time = time()
    tf.random.set_seed(42)
    np.random.seed(42)
    LOOKBACK = 15

    # Carga y preparación
    X, Y, dates, mean_val, std_val, n_features, df, target_idx = load_and_prepare_data(
        config.path, config.target_column, LOOKBACK
    )

    split_idx = int(len(dates) * (11 / 12))
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    # Construcción del modelo
    base_model = build_gru_model(config.architecture, LOOKBACK, n_features)
    model = MyModel(base_model)
    opt = Adam(learning_rate=1e-4, epsilon=1e-16)
    model.compile(optimizer=opt, loss=model.MyLoss, metrics=[model.MyLoss, model.MyMet])

    # Entrenamiento
    history = model.fit(
        X_train, Y_train,
        batch_size=50,
        epochs=config.epochs,
        verbose=0,
        validation_data=(X_test, Y_test)
    )

    # Predicción
    preds_all = model(X).numpy()
    preds_all_denorm = preds_all * std_val[target_idx] + mean_val[target_idx]
    Y_all_denorm = Y * std_val[target_idx] + mean_val[target_idx]
    dates_all = dates

    # Gráfico de ajuste
    plt.figure(figsize=(12, 6))
    plt.plot(dates_all, Y_all_denorm, 'b-', label="Real")
    plt.plot(dates_all, preds_all_denorm, 'r--', label="Predicho")
    split_date = dates_all.iloc[split_idx]
    plt.axvline(split_date, color='gray', linestyle='--', linewidth=1)
    plt.text(split_date, plt.ylim()[1] * 0.99, "← Entrenamiento | Predicción →",
             ha='center', va='top', color='gray', fontsize=10)
    plt.title(f"Ajuste y predicción ({config.target_column}) con GRU multivariada")
    plt.xlabel("Fecha")
    plt.ylabel(config.target_column)
    plt.legend()
    plt.tight_layout()
    graph_pred = os.path.join(GRAPH_PATH, f"prediccion_{config.target_column}.png")
    plt.savefig(graph_pred)
    plt.close()

    # Gráfico de pérdidas
    plt.figure(figsize=(8, 4))
    plt.semilogy(history.history['loss'], label="Train Loss (MSE)")
    plt.semilogy(history.history['val_loss'], label="Validation Loss")
    plt.xlabel("Época")
    plt.ylabel("Pérdida (log)")
    plt.legend()
    plt.tight_layout()
    graph_loss = os.path.join(GRAPH_PATH, f"perdida_{config.target_column}.png")
    plt.savefig(graph_loss)
    plt.close()

    elapsed = round(time() - start_time, 2)
    return {
        "modelo": "GRU multivariada",
        "configuracion_usada": CONFIG_FILE,
        "columna_objetivo": config.target_column,
        "arquitectura": config.architecture,
        "epochs": config.epochs,
        "n_features": n_features,
        "loss_final": float(history.history["loss"][-1]),
        "validation_loss_final": float(history.history["val_loss"][-1]),
        "graficos": {
            "prediccion": graph_pred,
            "perdidas": graph_loss
        },
        "tiempo_segundos": elapsed
    }
