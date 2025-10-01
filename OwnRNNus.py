import numpy as np
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
# Datos (ejemplo: 10 días del dólar en COP)
dolar_cop = np.array([4200, 4220, 4190, 4250, 4280,
                      4300, 4320, 4290, 4310, 4330], dtype=np.float32)

# Normalización
mean_val, std_val = np.mean(dolar_cop), np.std(dolar_cop)
data = (dolar_cop - mean_val) / std_val

# Dataset: usar x_t -> predecir x_{t+1}
T = 1
X_training = data[:-1].reshape(-1, T, 1)   # (N-1, time, features=1)
Y_training = data[1:].reshape(-1, 1)       # (N-1, 1)

# ---------------------------------------------------------------------------
# RNN definida con tf.Variable
# ---------------------------------------------------------------------------
class SimpleRNN(tf.keras.Model):
    def __init__(self, units, input_dim, output_dim):
        super(SimpleRNN, self).__init__()
        self.units = units
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Parámetros recurrentes
        self.Wx = tf.Variable(tf.random.normal([input_dim, units]), name="Wx")
        self.Wh = tf.Variable(tf.random.normal([units, units]), name="Wh")
        self.bh = tf.Variable(tf.zeros([units]), name="bh")

        # Parámetros de salida
        self.Wy = tf.Variable(tf.random.normal([units, output_dim]), name="Wy")
        self.by = tf.Variable(tf.zeros([output_dim]), name="by")

    @tf.function
    def call(self, x):
        # x: (batch, time, features)
        shape = tf.shape(x)
        batch_size = shape[0]
        T = shape[1]

        h = tf.zeros([batch_size, self.units])

        # AutoGraph convierte este for en un bucle válido de TF
        for t in range(T):
            xt = x[:, t, :]
            h = tf.tanh(tf.matmul(xt, self.Wx) + tf.matmul(h, self.Wh) + self.bh)

        y = tf.matmul(h, self.Wy) + self.by
        return y

    @staticmethod
    def MyLoss(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    @staticmethod
    def MyMet(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true/(y_true+0.1) - y_pred/(y_true+0.1)))


# Instancia del modelo con 3 nodos ocultos
model = SimpleRNN(units=3, input_dim=1, output_dim=1)

# ---------------------------------------------------------------------------
# Entrenamiento con compile/fit
# ---------------------------------------------------------------------------
opt = Adam(learning_rate=1e-2, epsilon=1e-16)

model.compile(optimizer=opt,
              loss=model.MyLoss,
              metrics=[model.MyLoss, model.MyMet])

history = model.fit(X_training, Y_training,
                    batch_size=2,
                    epochs=1000,
                    verbose=0)

# ---------------------------------------------------------------------------
# Resultados sobre datos de entrenamiento
# ---------------------------------------------------------------------------
preds = model(X_training).numpy()

# Desnormalizar
preds_denorm = preds * std_val + mean_val
Y_training_denorm = Y_training * std_val + mean_val

# Comparación entrenamiento
plt.figure(figsize=(8,5))
plt.plot(range(1, len(dolar_cop)), Y_training_denorm, 'bo-', label="Real")
plt.plot(range(1, len(dolar_cop)), preds_denorm, 'ro--', label="Predicho")
plt.xlabel("Día")
plt.ylabel("Dólar (COP)")
plt.legend()
plt.show()

# ---------------------------------------------------------------------------
# Predicción hacia adelante (5 días)
# ---------------------------------------------------------------------------
last_val = data[-1].reshape(1, 1, 1)  # último valor conocido (día 10)
future_preds = []
horizon = 5

current_input = last_val
for _ in range(horizon):
    next_val = model(current_input).numpy()
    future_preds.append(next_val[0,0])
    current_input = next_val.reshape(1, 1, 1)  # retroalimentar la predicción

# Desnormalizar las predicciones
future_preds = np.array(future_preds) * std_val + mean_val

# Gráfica con los días futuros
plt.figure(figsize=(8,5))
plt.plot(range(1, len(dolar_cop)+1), dolar_cop, 'bo-', label="Serie real (10 días)")
plt.plot(range(1, len(dolar_cop)), preds_denorm, 'ro--', label="Predicción entrenamiento")
plt.plot(range(len(dolar_cop)+1, len(dolar_cop)+horizon+1), future_preds, 'gs--', label="Predicción futura")
plt.xlabel("Día")
plt.ylabel("Dólar (COP)")
plt.legend()
plt.show()

print(f"Tiempo total: {time() - start_time:.2f} segundos")

