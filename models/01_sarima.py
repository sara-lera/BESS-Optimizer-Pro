import pandas as pd
import numpy as np
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def main():
    print("--- Modelo SARIMA Univariante / Multivariante ---")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "raw", "dataset_multivariante_completo.csv")
    
    if not os.path.exists(data_path):
        print("Dataset no encontrado. Debes correr primero data_ingestion.py")
        return
        
    df = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
    df = df.dropna()
    
    # Vamos a predecir precio_mwh
    n_test = 96 * 2 # 2 dias a 15 min
    train = df.iloc[:-n_test]
    test = df.iloc[-n_test:]
    
    print(f"Train size: {len(train)}, Test size: {len(test)}")
    
    # Exogenous variables: la demanda
    exog_train = train[['demanda']]
    exog_test = test[['demanda']]
    
    print("Ajustando SARIMA(1,1,1)...")
    # Nota: Si añades estacionalidad (ej. seasonal_order=(1,0,1,96)),
    # puede tardar mucho en entrenar, por eso se mantiene simple
    model = SARIMAX(train['precio_mwh'], 
                    exog=exog_train,
                    order=(1, 1, 1))
    
    fit_model = model.fit(disp=False)
    
    print("Prediciendo...")
    preds = fit_model.predict(start=len(train), end=len(train)+len(test)-1, exog=exog_test)
    preds.index = test.index
    
    mae = mean_absolute_error(test['precio_mwh'], preds)
    rmse = np.sqrt(mean_squared_error(test['precio_mwh'], preds))
    print(f"SARIMA - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(test.index, test['precio_mwh'], label='Real')
    plt.plot(test.index, preds, label='SARIMA Predict', linestyle='dashed')
    plt.title("Predicción SARIMA (2026 - 15 min)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(script_dir, "sarima_plot.png"))
    print("Guardado en sarima_plot.png")

if __name__ == "__main__":
    main()
