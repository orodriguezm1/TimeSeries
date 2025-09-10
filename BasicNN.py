import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from time import time

# ---------------------------------------------------------------------------
# Inicialización
# ---------------------------------------------------------------------------
start_time = time()
tf.random.set_seed(42)   # Reproducibilidad en TensorFlow
np.random.seed(42)       # Reproducibilidad en NumPy

# ---------------------------------------------------------------------------
# Definición del problema
# ---------------------------------------------------------------------------

def FP(x):
    """Función objetivo a aproximar: f(x) = x^2"""
    return x**2

# Rango de datos
min_x, max_x = 0.05, 0.5
samples, samples_val = 10000, 1000

# Datos de entrenamiento
X_training = tf.random.uniform((samples, 1), minval=min_x, maxval=max_x)
Y_training = FP(X_training)

# Datos de validación (en malla regular)
X_val = np.linspace(min_x, max_x, samples_val).reshape(-1, 1)
Y_val = FP(X_val)

# ---------------------------------------------------------------------------
# Construcción de la red neuronal
# ---------------------------------------------------------------------------

inp, out = 1, 1    # Dimensiones de entrada y salida
nod, lay = 10, 4   # Nodos y capas ocultas

# Arquitectura definida dinámicamente
nodes_NN = [nod] * (lay - 1) + [nod, out]
activations = ['tanh'] * (lay - 1) + ['softplus', 'linear']

# Modelo secuencial
model_ = Sequential(name="Feedforward_NN")
for i, (n, act) in enumerate(zip(nodes_NN, activations)):
    if i == 0:  # Capa de entrada
        model_.add(Dense(n, activation=act, input_shape=(inp,),
                         use_bias=False,
                         kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                         name="Input_layer"))
    elif i == len(nodes_NN) - 1:  # Capa de salida
        model_.add(Dense(n, activation=act,
                         use_bias=True,
                         kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                         name="Output_layer"))
    else:  # Capas ocultas
        model_.add(Dense(n, activation=act,
                         use_bias=False,
                         kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                         name=f"Hidden_layer_{i}"))

# ---------------------------------------------------------------------------
# Modelo personalizado
# ---------------------------------------------------------------------------

class MyModel(tf.keras.Model):
    """Modelo wrapper para añadir funciones de pérdida y métricas personalizadas."""
    def __init__(self, base_model, name=None):
        super(MyModel, self).__init__(name=name)
        self.loc_net = base_model

    def call(self, x):
        return self.loc_net(x)

    @staticmethod
    def MyLoss(y_true, y_pred):
        """Error cuadrático medio (MSE)."""
        return tf.reduce_mean(tf.math.squared_difference(y_true, y_pred))

    @staticmethod
    def MyMet(y_true, y_pred):
        """
        Métrica modificada:
        compara predicciones y verdaderos escalados con (y + 0.1).
        """
        return tf.reduce_mean(tf.math.squared_difference(y_true / (y_true + 0.1),
                                                         y_pred / (y_true + 0.1)))

# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

model = MyModel(model_)
opt = Adam(learning_rate=1e-3, epsilon=1e-16)

model.compile(optimizer=opt,
              loss=model.MyLoss,
              metrics=[model.MyLoss, model.MyMet])

history = model.fit(X_training, Y_training,
                    batch_size=1000,
                    epochs=1000,
                    validation_data=(X_val, Y_val),
                    verbose=0)  # verbose=0 para entrenamiento silencioso

# ---------------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------------

# Predicciones sobre el conjunto de validación
NN_val = model(X_val).numpy()

# Gráfico comparando función exacta vs. aproximada
plt.figure(figsize=(8, 6))
plt.plot(X_val, Y_val, 'bo', alpha=0.5, label="Exact")
plt.plot(X_val, NN_val, 'ro', alpha=0.5, label="Estimation")
plt.xlabel('x', fontsize=20)
plt.ylabel('u(x)', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.show()

# Gráfico de la función de pérdida
plt.figure(figsize=(8, 6))
plt.semilogy(history.history['loss'], label="Train Loss")
plt.semilogy(history.history['val_loss'], label="Validation Loss")
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Loss (log scale)', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.show()

print(f"Tiempo total: {time() - start_time:.2f} segundos")
