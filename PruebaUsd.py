import yfinance as yf
import pandas as pd
from datetime import date, timedelta

# 📅 Fechas del último mes
end = date.today()
start = end - timedelta(days=30)

print(f"Descargando datos desde {start} hasta {end}...\n")

def descargar_cierre(ticker, nombre):
    """Descarga el precio de cierre y aplana columnas si es necesario"""
    df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
    # Aplanar columnas si son multinivel
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    # Buscar la columna de cierre
    close_cols = [c for c in df.columns if 'Close' in c]
    if not close_cols:
        raise ValueError(f"No se encontró columna 'Close' en {ticker}")
    df = df[[close_cols[0]]].rename(columns={close_cols[0]: nombre})
    return df

# 💵 Tipo de cambio USD/COP
usd_cop = descargar_cierre("USDCOP=X", "USD_COP")

# 🛢️ Precio del Brent (USD)
brent = descargar_cierre("BZ=F", "Brent_USD")

# 📈 Tasa de bonos de EE. UU. (proxy tasa FED)
fed_rate = descargar_cierre("^IRX", "FED_RATE")

# ⚡ Índice de volatilidad VIX
vix = descargar_cierre("^VIX", "VIX")

# 🏛️ S&P500 (referencia global)
sp500 = descargar_cierre("^GSPC", "SP500")

# 🔗 Combinar todo por fecha
df = usd_cop.merge(brent, left_index=True, right_index=True, how="inner")
df = df.merge(fed_rate, left_index=True, right_index=True, how="inner")
df = df.merge(vix, left_index=True, right_index=True, how="inner")
df = df.merge(sp500, left_index=True, right_index=True, how="inner")

# 💰 Agregar columna: Precio del barril en COP
df["Brent_COP"] = df["USD_COP"] * df["Brent_USD"]

# 📊 Reordenar columnas
df = df[["USD_COP", "Brent_USD", "Brent_COP", "FED_RATE", "VIX", "SP500"]]
df.index.name = "Fecha"

# 💾 Guardar resultados
df.to_csv("indicadores_usd_petroleo.csv")

print("✅ Archivo 'indicadores_usd_petroleo.csv' generado con éxito.\n")
print(df.tail())
