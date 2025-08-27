# ================================================================
# COMPARATIVA: RNN/LSTM (TensorFlow) vs Modelos clásicos
#   Parte A (Univariante, CO2 mensual): ARIMA vs SimpleRNN vs LSTM
#   Parte B (Multivariado, Macro trimestral): VAR vs LSTM multivariable
# ================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR

# ---------- utilidades ----------
def rmse(y_true, y_pred): return np.sqrt(mean_squared_error(y_true, y_pred))
def mae(y_true, y_pred):  return mean_absolute_error(y_true, y_pred)

def make_seq_univar(arr, look_back=12):
    """arr: (N,1) -> X: (N-look_back, look_back, 1), y: (N-look_back,)"""
    X, y = [], []
    for i in range(look_back, len(arr)):
        X.append(arr[i-look_back:i, 0])
        y.append(arr[i, 0])
    X, y = np.array(X), np.array(y)
    return X.reshape((-1, look_back, 1)), y

def make_seq_multivar(X2d, y1d, look_back=8):
    """X2d:(N,F) y1d:(N,) -> X:(N-L, L, F), y:(N-L,)"""
    Xs, ys = [], []
    for t in range(look_back, len(X2d)):
        Xs.append(X2d[t-look_back:t, :])
        ys.append(y1d[t])
    return np.array(Xs), np.array(ys)

tf.random.set_seed(123); np.random.seed(123)

# ================================================================
# PARTE A — UNIVARIANTE: CO2 mensual (ARIMA vs SimpleRNN vs LSTM)
# ================================================================
# 1) Datos (CO2 desde statsmodels) y split temporal
co2 = sm.datasets.co2.load_pandas().data.rename(columns={"co2":"y"})
co2.index = pd.to_datetime(co2.index)
s = co2["y"].resample("ME").mean().interpolate("linear").dropna().to_frame()
splitA = int(len(s)*0.85)
trainA = s.iloc[:splitA].copy()
testA  = s.iloc[splitA:].copy()

# 2) ARIMA (clásico) — pronóstico h=len(testA)
#    (Simple: order fijo; en la práctica probarías auto_arima/SARIMA)
arima = ARIMA(trainA["y"], order=(5,1,0))
arima_fit = arima.fit()
fc_arima = arima_fit.forecast(steps=len(testA))  # serie pronosticada
rmse_arima = rmse(testA["y"].values, fc_arima.values)
mae_arima  = mae(testA["y"].values, fc_arima.values)

# 3) Deep Learning (RNN/LSTM) con ventanas look_back
look_backA = 12
scA = MinMaxScaler()
y_train_sc = scA.fit_transform(trainA[["y"]].values)
XtrA, ytrA  = make_seq_univar(y_train_sc, look_backA)

# contexto: usar EXACTAMENTE look_back puntos previos de train
y_test_sc = scA.transform(testA[["y"]].values)
y_concat  = np.vstack([y_train_sc[-look_backA:], y_test_sc])
XtA, ytA  = make_seq_univar(y_concat, look_backA)     # produce len(testA) targets
# ytA está en escala [0,1] y corresponde 1:1 con testA

# 3.1) SimpleRNN
rnnA = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(look_backA,1)),
    tf.keras.layers.SimpleRNN(64, return_sequences=False),
    tf.keras.layers.Dense(1)
])
rnnA.compile(optimizer='adam', loss='mse')
rnnA.fit(XtrA, ytrA, epochs=80, batch_size=32, verbose=0, validation_split=0.2)
pred_rnnA_sc = rnnA.predict(XtA, verbose=0)
pred_rnnA    = scA.inverse_transform(pred_rnnA_sc).ravel()
rmse_rnnA = rmse(testA["y"].values, pred_rnnA)
mae_rnnA  = mae(testA["y"].values, pred_rnnA)

# 3.2) LSTM
lstmA = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(look_backA,1)),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1)
])
lstmA.compile(optimizer='adam', loss='mse')
lstmA.fit(XtrA, ytrA, epochs=80, batch_size=32, verbose=0, validation_split=0.2)
pred_lstmA_sc = lstmA.predict(XtA, verbose=0)
pred_lstmA    = scA.inverse_transform(pred_lstmA_sc).ravel()
rmse_lstmA = rmse(testA["y"].values, pred_lstmA)
mae_lstmA  = mae(testA["y"].values, pred_lstmA)

# 4) Gráfico univariante
plt.figure(figsize=(12,5))
plt.plot(trainA.index, trainA["y"], label="Train CO₂")
plt.plot(testA.index,  testA["y"],  label="Test CO₂", color="black")
plt.plot(testA.index,  fc_arima.values, "--", label=f"ARIMA | RMSE={rmse_arima:.2f}")
plt.plot(testA.index,  pred_rnnA, "--",  label=f"SimpleRNN | RMSE={rmse_rnnA:.2f}")
plt.plot(testA.index,  pred_lstmA, "--", label=f"LSTM | RMSE={rmse_lstmA:.2f}")
plt.title("CO₂ mensual — ARIMA vs SimpleRNN vs LSTM")
plt.grid(alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

print("=== PARTE A: CO₂ (univariante) ===")
print(f"ARIMA     -> RMSE {rmse_arima:.3f} | MAE {mae_arima:.3f}")
print(f"SimpleRNN -> RMSE {rmse_rnnA:.3f} | MAE {mae_rnnA:.3f}")
print(f"LSTM      -> RMSE {rmse_lstmA:.3f} | MAE {mae_lstmA:.3f}")

# ================================================================
# PARTE B — MULTIVARIADO: Macro trimestral
#   Objetivo: predecir 'infl' (inflación) con VAR vs LSTM multivar
# ================================================================
# 1) Datos macro (statsmodels.macrodata)
macro = sm.datasets.macrodata.load_pandas().data.copy()
periods = pd.period_range(
    start=f"{int(macro['year'].iloc[0])}Q{int(macro['quarter'].iloc[0])}",
    periods=len(macro), freq='Q'
)
df = macro.set_index(periods).drop(columns=['year','quarter'])
df.index = df.index.to_timestamp()

target = "infl"
features = ['realgdp','realcons','realinv','realdpi','cpi','m1','tbilrate','unemp','pop','realint']

splitB = int(len(df)*0.85)
trainB = df.iloc[:splitB].copy()
testB  = df.iloc[splitB:].copy()

# 2) VAR (clásico)
var = VAR(trainB[[target]+features])
res_var = var.fit(maxlags=4, ic='aic')  # deja que elija lags hasta 4 (o fija 4)
# forecast necesita los últimos k_obs (lags) de train:
k = res_var.k_ar
last_obs = trainB[[target]+features].values[-k:]
fc_var = res_var.forecast(y=last_obs, steps=len(testB))  # matriz con todas las series
# Extrae la columna de 'infl' (target es la primera si la pasamos así)
colnames = [target] + features
pred_var_infl = pd.DataFrame(fc_var, index=testB.index, columns=colnames)[target].values
rmse_var = rmse(testB[target].values, pred_var_infl)
mae_var  = mae(testB[target].values, pred_var_infl)

# 3) LSTM multivariada (usa features como entrada y predice infl)
look_backB = 8
X_train = trainB[features].values
y_train = trainB[target].values
X_test  = testB[features].values
y_test  = testB[target].values

# Escalado separado para X e y
scX = StandardScaler()
scY = StandardScaler()
Xtr_s = scX.fit_transform(X_train)
Xte_s = scX.transform(X_test)
ytr_s  = scY.fit_transform(y_train.reshape(-1,1)).ravel()
yte_s  = scY.transform(y_test.reshape(-1,1)).ravel()

# Secuencias 3D — train
Xtr3, ytr3 = make_seq_multivar(Xtr_s, ytr_s, look_backB)
# Secuencias 3D — test con EXACTAMENTE 'look_back' de contexto desde train
Xcat = np.vstack([Xtr_s[-look_backB:], Xte_s])
ycat = np.hstack([ytr_s[-look_backB:], yte_s])
Xte3, yte3 = make_seq_multivar(Xcat, ycat, look_backB)   # yte3 ~ yte_s

# Modelo LSTM
tf.random.set_seed(123)
nF = Xtr3.shape[-1]
lstmB = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(look_backB, nF)),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1)
])
lstmB.compile(optimizer='adam', loss='mse')
lstmB.fit(Xtr3, ytr3, epochs=100, batch_size=16, verbose=0, validation_split=0.2)

predB_s = lstmB.predict(Xte3, verbose=0).ravel()
predB   = scY.inverse_transform(predB_s.reshape(-1,1)).ravel()  # escala original

# 4) Métricas y gráfico multivariado (alineado 1:1)
rmse_lstmB = rmse(y_test, predB)
mae_lstmB  = mae(y_test, predB)

plt.figure(figsize=(12,5))
plt.plot(testB.index, y_test, label="Real infl (test)", color="black")
plt.plot(testB.index, pred_var_infl, "--", label=f"VAR | RMSE={rmse_var:.2f}")
plt.plot(testB.index, predB, "--", label=f"LSTM multivar | RMSE={rmse_lstmB:.2f}")
plt.title("Inflación trimestral — VAR vs LSTM multivariable")
plt.grid(alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

print("\n=== PARTE B: Macro (multivariado) ===")
print(f"VAR            -> RMSE {rmse_var:.3f} | MAE {mae_var:.3f}")
print(f"LSTM multivar  -> RMSE {rmse_lstmB:.3f} | MAE {mae_lstmB:.3f}")

