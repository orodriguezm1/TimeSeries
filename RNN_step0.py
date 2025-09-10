import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# =================================
# 1. Generar datos simulados
# =================================
T = 100
t = np.arange(1, T + 1)
x_data = np.sin(0.2 * t) + 0.1 * np.random.randn(T)  # señal senoidal con ruido

# dataset: predecir el siguiente valor
X = x_data[:-1]  # entradas hasta T-1
Y = x_data[1:]  # targets desplazados

X = X.reshape(-1, 1, 1).astype(np.float32)  # (batch, time, features)
Y = Y.reshape(-1, 1).astype(np.float32)  # (batch, output)

# =================================
# 2. Definir pesos para 2 capas
#    Cada capa con 3 nodos ocultos
# =================================
m1, m2 = 10, 10  # nodos ocultos capa 1 y capa 2

# Capa 1
W_in1 = tf.Variable(tf.random.normal([1, m1], stddev=0.5))
U1 = tf.Variable(tf.random.normal([m1, m1], stddev=0.5))
b1 = tf.Variable(tf.zeros([m1]))

# Capa 2
W_in2 = tf.Variable(tf.random.normal([m1, m2], stddev=0.5))
U2 = tf.Variable(tf.random.normal([m2, m2], stddev=0.5))
b2 = tf.Variable(tf.zeros([m2]))

# Salida
W_y = tf.Variable(tf.random.normal([m2, 1], stddev=0.5))
b_y = tf.Variable(tf.zeros([1]))


# =================================
# 3. Forward paso a paso
# =================================
@tf.function
def rnn_forward(x_seq):
    batch_size = tf.shape(x_seq)[0]
    h1 = tf.zeros([batch_size, m1])  # estado inicial capa 1
    h2 = tf.zeros([batch_size, m2])  # estado inicial capa 2

    for t in range(x_seq.shape[1]):
        x_t = x_seq[:, t, :]  # (batch, features)
        h1 = tf.tanh(tf.matmul(x_t, W_in1) + tf.matmul(h1, U1) + b1)
        h2 = tf.tanh(tf.matmul(h1, W_in2) + tf.matmul(h2, U2) + b2)

    y_hat = tf.matmul(h2, W_y) + b_y
    return y_hat


# =================================
# 4. Definir modelo Keras
# =================================
class TwoLayerRNN(tf.keras.Model):
    def call(self, inputs):
        return rnn_forward(inputs)


model = TwoLayerRNN()

# =================================
# 5. Entrenamiento
# =================================
LR = [0.1, 0.01]
EP = [2000, 3000]
for i in range(len(LR)):
    model.compile(optimizer=tf.keras.optimizers.Adam(LR[i]), loss='mse')
    history = model.fit(X, Y, epochs=EP[i], verbose=0)

# =================================
# 6. Predicción futura (101–110)
# =================================
future_steps = 10
x_full = list(x_data)

preds_caseA = []
preds_caseB1 = []
preds_caseB2 = []

for step in range(future_steps):
    # Caso A: tenemos el input real futuro (usamos senoide verdadera)
    true_x = np.sin(0.2 * (T + step + 1))  # valor real futuro
    xA = np.array([[[true_x]]], dtype=np.float32)
    yA = model(xA).numpy().flatten()[0]
    preds_caseA.append(yA)

    # Caso B1: persistencia
    last_x = x_full[-1]
    xB1 = np.array([[[last_x]]], dtype=np.float32)
    yB1 = model(xB1).numpy().flatten()[0]
    preds_caseB1.append(yB1)
    x_full.append(last_x)  # persistencia

    # Caso B2: autoregresivo (usar predicción previa como input)
    if step == 0:
        inputB2 = x_data[-1]  # primer input = último real observado
    else:
        inputB2 = preds_caseB2[-1]
    xB2 = np.array([[[inputB2]]], dtype=np.float32)
    yB2 = model(xB2).numpy().flatten()[0]
    preds_caseB2.append(yB2)

# =================================
# 7. Gráficas
# =================================

# =================================
# 7. Gráficas combinadas
# =================================

# Predicciones "in-sample" (ajuste sobre los datos usados en entrenamiento)
Y_hat_train = model(X).numpy().flatten()

plt.figure(figsize=(12,6))
plt.plot(range(1, T+1), x_data, 'k', label="Datos observados")
plt.plot(range(2, T+1), Y_hat_train, 'r--', label="Ajuste del modelo (train)")

# Predicciones futuras
plt.plot(range(T+1, T+future_steps+1), preds_caseA, 'o--', label="Predicción Caso A (x_t real)")
plt.plot(range(T+1, T+future_steps+1), preds_caseB1, 's--', label="Predicción Caso B1 (persistencia)")
plt.plot(range(T+1, T+future_steps+1), preds_caseB2, 'd--', label="Predicción Caso B2 (autoregresivo)")

plt.axvline(T, color="gray", linestyle=":", alpha=0.7)  # línea que separa train vs predicción
plt.legend()
plt.title("Ajuste de la RNN y predicciones futuras")
plt.xlabel("Tiempo")
plt.ylabel("Valor")
plt.show()


# Función de costo (escala log)
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'])
plt.yscale("log")
plt.xlabel("Época")
plt.ylabel("Loss (MSE, log)")
plt.title("Evolución de la función de costo (escala log)")
plt.show()
