import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

def load_and_prepare_data(file_path):
    print(f"Cargando datos desde {file_path}")
    df = pd.read_csv(file_path, parse_dates=['datetime'], index_col='datetime')
    
    # Nos centramos en predecir precio_mwh
    if 'precio_mwh' not in df.columns:
        raise ValueError("La columna 'precio_mwh' no está en el dataset")
        
    df = df.dropna() # por si quedaron nulos

    # Crear características basadas en tiempo (para los modelos de ML avanzados)
    df['hora'] = df.index.hour
    df['dia_semana'] = df.index.dayofweek
    df['mes'] = df.index.month
    
    # Crear características desfasadas (lags)
    # Por ejemplo, el precio hace 24 horas y hace 1 hora
    df['precio_lag_1h'] = df['precio_mwh'].shift(4)  # 1h son 4 lags de 15 min
    df['precio_lag_24h'] = df['precio_mwh'].shift(96) # 24h son 96 lags de 15 min
    
    df = df.dropna()
    return df

def train_sarima(train, test):
    print("\n--- Entrenando Modelo SARIMA ---")
    # Para ser rápidos, usamos un orden básico, idealmente se busca con auto_arima
    # Usamos exógenas: demanda real
    exog_train = train[['demanda_real']] if 'demanda_real' in train.columns else None
    exog_test = test[['demanda_real']] if 'demanda_real' in test.columns else None
    
    model = SARIMAX(train['precio_mwh'], 
                    exog=exog_train, 
                    order=(1, 1, 1), 
                    seasonal_order=(0, 0, 0, 0)) # No estacionalidad para acelerar, o (1,0,1,96) si computa
    
    sarima_fit = model.fit(disp=False)
    print("Prediciendo...")
    preds = sarima_fit.predict(start=len(train), end=len(train) + len(test) - 1, exog=exog_test)
    preds.index = test.index
    
    mae = mean_absolute_error(test['precio_mwh'], preds)
    rmse = np.sqrt(mean_squared_error(test['precio_mwh'], preds))
    print(f"SARIMA MAE: {mae:.2f}")
    print(f"SARIMA RMSE: {rmse:.2f}")
    
    return preds

def train_random_forest(train, test):
    print("\n--- Entrenando RandomForest (Modelo Avanzado de ML / Tabular) ---")
    features = [c for c in train.columns if c != 'precio_mwh']
    
    X_train, y_train = train[features], train['precio_mwh']
    X_test, y_test = test[features], test['precio_mwh']
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    print("Prediciendo...")
    preds = rf.predict(X_test)
    preds = pd.Series(preds, index=test.index)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"Random Forest MAE: {mae:.2f}")
    print(f"Random Forest RMSE: {rmse:.2f}")
    
    return preds, rf

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.normpath(os.path.join(script_dir, "..", "data", "raw", "dataset_multivariante_completo.csv"))
    
    df = load_and_prepare_data(file_path)
    
    # Train-test split (p.ej. últimos 2 días para test)
    # Dado que es a 15min, 2 días son 192 pasos
    n_test = 192
    train = df.iloc[:-n_test]
    test = df.iloc[-n_test:]
    
    print(f"Train size: {len(train)}, Test size: {len(test)}")
    
    # 1. SARIMA
    preds_sarima = train_sarima(train, test)
    
    # 2. Random Forest Regressor
    preds_rf, model_rf = train_random_forest(train, test)
    
    # Gráfica
    plt.figure(figsize=(14, 6))
    plt.plot(test.index, test['precio_mwh'], label='Real', color='black', linewidth=2)
    plt.plot(test.index, preds_sarima, label='SARIMA', alpha=0.8)
    plt.plot(test.index, preds_rf, label='Random Forest', alpha=0.8)
    plt.title('Comparativa Predicción: Precio de la Electricidad (MWh)')
    plt.legend()
    plt.grid(True)
    
    out_img = os.path.join(script_dir, "..", "predicciones.png")
    plt.savefig(out_img)
    plt.close()
    print(f"\n¡Completado! Gráfico de predicción guardado en {out_img}")
