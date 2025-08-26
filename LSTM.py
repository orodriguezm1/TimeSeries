import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import yfinance as yf

# ===== 1) Descargar datos (S&P 500, '^GSPC') y agregarlos a mensual =====
data = yf.download("^GSPC", start="1990-01-01", auto_adjust=True, progress=False)
# Usamos el cierre y lo transformamos a fin de mes
df = data[["Close"]].resample("ME").last().rename(columns={"Close": "y"}).dropna()

# ===== 2) Split train/test =====
split_date = df.index[int(len(df)*0.85)]
train = df.loc[:split_date]
test  = df.loc[split_date:]

# ===== 3) Escalado =====
scaler = MinMaxScaler()
series_tr = scaler.fit_transform(train[["y"]].values)

# ===== 4) Secuencias =====
look_back = 12
def make_sequences(arr, look_back=12):
    X, Y = [], []
    for i in range(look_back, len(arr)):
        X.append(arr[i-look_back:i, 0])
        Y.append(arr[i, 0])
    X = np.array(X); Y = np.array(Y)
    X = X.reshape((X.shape[0], look_back, 1))
    return X, Y

X_train, Y_train = make_sequences(series_tr, look_back=look_back)

# ===== 5) Modelo LSTM =====
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(look_back, 1)),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=0)

# ===== 6) Test =====
series_te = scaler.transform(test[["y"]].values)
X_test, Y_test = make_sequences(series_te, look_back=look_back)

# ===== 7) Predicción y RMSE =====
Y_pred = model.predict(X_test, verbose=0)
Y_pred_inv = scaler.inverse_transform(Y_pred)
Y_true_inv = test["y"].values[look_back:]
rmse_lstm = np.sqrt(mean_squared_error(Y_true_inv, Y_pred_inv))
print(f"RMSE LSTM (S&P500 mensual): {rmse_lstm:.3f}")
