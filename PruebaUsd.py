import yfinance as yf
import warnings
warnings.filterwarnings("ignore", message=".*verbose is deprecated.*")
import pandas as pd
from datetime import date, timedelta
from statsmodels.tsa.stattools import grangercausalitytests


# ============================================================
# CONFIGURACIÓN
# ============================================================

# Rango temporal: último año de datos diarios
end = date.today()
start = end - timedelta(days=365)
print(f"Descargando datos desde {start} hasta {end}...\n")


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def descargar_cierre(tickers):
    """
    Descarga precios de cierre ajustados de múltiples activos desde Yahoo Finance.
    Devuelve un DataFrame con una columna por activo, indexado por fecha.
    """
    datos = {}
    for nombre, ticker in tickers.items():
        df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True, progress=False)

        # Si las columnas son multinivel, se aplanan
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]

        # Selecciona la columna de cierre
        col_cierre = [c for c in df.columns if "Close" in c][0]
        datos[nombre] = df[col_cierre]

    return pd.DataFrame(datos)


def test_granger(df, variable_dependiente, variable_independiente, maxlag=5):
    """
    Ejecuta un test de causalidad de Granger para verificar si una variable
    (variable_independiente) tiene poder predictivo sobre otra (variable_dependiente).

    Retorna el valor p mínimo obtenido entre los rezagos analizados.
    """
    resultado = grangercausalitytests(df[[variable_dependiente, variable_independiente]],
                                      maxlag=maxlag, verbose=False)
    p_values = [resultado[i + 1][0]['ssr_ftest'][1] for i in range(maxlag)]
    return min(p_values)


# ============================================================
# DESCARGA DE DATOS
# ============================================================

tickers = {
    "USD_COP": "USDCOP=X",   # Tipo de cambio Peso Colombiano / Dólar
    "Brent_USD": "BZ=F",     # Precio del petróleo Brent en USD
    "FED_RATE": "^IRX",      # Tasa de bonos del Tesoro (proxy FED)
    "VIX": "^VIX",           # Índice de volatilidad global
    "SP500": "^GSPC"         # Índice S&P 500
}

df = descargar_cierre(tickers)

# ============================================================
# TRANSFORMACIONES Y CÁLCULOS
# ============================================================

# Precio del barril en pesos colombianos
df["Brent_COP"] = df["USD_COP"] * df["Brent_USD"]

# Reordenar columnas y establecer índice
df = df[["USD_COP", "Brent_COP", "FED_RATE", "VIX", "SP500"]]
df.index.name = "Fecha"

# Guardar los resultados
df.to_csv("indicadores_usd_petroleo.csv")
print("Archivo 'indicadores_usd_petroleo.csv' generado con éxito.\n")
print(df.tail())


# ============================================================
# TEST DE CAUSALIDAD DE GRANGER
# ============================================================

# Diferencias para estacionarizar las series
data_diff = df[["USD_COP", "Brent_COP", "FED_RATE", "VIX", "SP500"]].dropna().diff().dropna()

# Pares de variables para probar causalidad
pares = [
    ("USD_COP", "Brent_COP"),  # ¿El petróleo predice el dólar?
    ("USD_COP", "FED_RATE"),   # ¿La tasa FED predice el dólar?
    ("USD_COP", "VIX"),        # ¿El VIX predice el dólar?
    ("USD_COP", "SP500"),        # ¿El SP500 predice el dólar?
]

causales_significativas = []
print("\nResultados de causalidad de Granger (p-valores mínimos):\n")
for dependiente, independiente in pares:
    p_value = test_granger(data_diff, dependiente, independiente, maxlag=5)
    es_causal = p_value < 0.05
    resultado = "Existe causalidad (p < 0.05)" if es_causal else "No se encontró causalidad"
    print(f"{independiente} → {dependiente}: p = {p_value:.4f} | {resultado}")

    if es_causal:
        causales_significativas.append((dependiente, independiente))

# ============================================================
# GUARDAR CSV CON VARIABLES CAUSALES
# ============================================================

if causales_significativas:
    # Identificar todas las variables que participan en relaciones significativas
    variables_significativas = set()
    for dep, indep in causales_significativas:
        variables_significativas.update([dep, indep])

    # Filtrar el DataFrame original para mantener solo esas columnas
    df_causales = df[list(variables_significativas)]
    df_causales.to_csv("indicadores_causales.csv")

    print("\nSe detectaron relaciones causales significativas.")
    print(f"Variables incluidas en 'indicadores_causales.csv': {', '.join(variables_significativas)}")
else:
    print("\nNo se detectaron relaciones causales significativas. No se generó archivo adicional.")