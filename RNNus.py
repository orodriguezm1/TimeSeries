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
# Datos (ejemplo: 10 días de cotización del dólar en COP)
# Sustituye esta lista por los valores reales
dolar_cop = np.array([4200, 4220, 4190, 4250, 4280, 4300, 4320, 4290, 4310, 4330], dtype=np.float32)

# Normalización
mean_val, std_val = np.mean(dolar_cop), np.std(dolar_cop)
data = (dolar_cop - mean_val) / std_val

# Construcción de dataset (usar x_t -> predecir x_{t+1})
X_training = data[:-1].reshape(-1, 1, 1)   # (N-1, time=1, features=1)
Y_training = data[1:].reshape(-1, 1)       # (N-1, 1)

# ---------------------------------------------------------------------------
# Construcción de la RNN
# ---------------------------------------------------------------------------
model_ = tf.keras.Sequential(name="Simple_RNN")
model_.add(tf.keras.layers.SimpleRNN(units=3, activation="tanh",
                                     input_shape=(1,1),  # secuencia de longitud 1, 1 feature
                                     use_bias=True,
                                     kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                                     name="RNN_layer"))
model_.add(tf.keras.layers.Dense(1, activation="linear", name="Output_layer"))

# Modelo wrapper con funciones de pérdida y métrica personalizadas
class MyModel(tf.keras.Model):
    def __init__(self, base_model, name=None):
        super(MyModel, self).__init__(name=name)
        self.loc_net = base_model

    def call(self, x):
        return self.loc_net(x)

    @staticmethod
    def MyLoss(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    @staticmethod
    def MyMet(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true/(y_true+0.1) - y_pred/(y_true+0.1)))

# Instanciamos el wrapper
model = MyModel(model_)

# Compilamos con Adam
opt = Adam(learning_rate=1e-2, epsilon=1e-16)
model.compile(optimizer=opt,
              loss=model.MyLoss,
              metrics=[model.MyLoss, model.MyMet])

# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
history = model.fit(X_training, Y_training,
                    batch_size=2,
                    epochs=1000,
                    verbose=0)

# ---------------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------------
preds = model(X_training).numpy()

# Desnormalizar
preds_denorm = preds * std_val + mean_val
Y_training_denorm = Y_training * std_val + mean_val

# Comparación
plt.figure(figsize=(8,5))
plt.plot(range(1, len(dolar_cop)), Y_training_denorm, 'bo-', label="Real")
plt.plot(range(1, len(dolar_cop)), preds_denorm, 'ro--', label="Predicho")
plt.xlabel("Día")
plt.ylabel("Dólar (COP)")
plt.legend()
plt.show()

# Pérdida
plt.figure(figsize=(8,5))
plt.semilogy(history.history['loss'], label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE (log scale)")
plt.legend()
plt.show()

print(f"Tiempo total: {time() - start_time:.2f} segundos")
