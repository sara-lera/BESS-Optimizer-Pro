import pandas as pd
import numpy as np
import os
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def main():
    print("--- Modelo VARIMA (Vectores Autorregresivos) ---")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "raw", "dataset_multivariante_completo.csv")
    
    if not os.path.exists(data_path):
        print("Dataset no encontrado.")
        return
        
    df = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
    df = df.dropna()
    
    # VARIMA modela conjuntamente multiples variables endogenas.
    # Elegimos precio y demanda
    target_cols = ['precio_mwh', 'demanda_real']
    
    n_test = 96 * 2 # 2 dias
    train = df.iloc[:-n_test][target_cols]
    test = df.iloc[-n_test:][target_cols]
    
    print("Ajustando VARMAX(1,0)...")
    model = VARMAX(train, order=(1, 0), enforce_stationarity=False)
    fit_model = model.fit(disp=False)
    
    print("Prediciendo...")
    preds = fit_model.forecast(steps=n_test)
    preds.index = test.index
    
    # Metricas solo para el precio
    mae = mean_absolute_error(test['precio_mwh'], preds['precio_mwh'])
    rmse = np.sqrt(mean_squared_error(test['precio_mwh'], preds['precio_mwh']))
    print(f"VARIMA (Precio) - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(test.index, test['precio_mwh'], label='Real')
    plt.plot(test.index, preds['precio_mwh'], label='VARIMA Predict', linestyle='dashed')
    plt.title("Predicción VARIMA Conjunta (Precio y Demanda)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(script_dir, "varima_plot.png"))
    print("Guardado en varima_plot.png")

if __name__ == "__main__":
    main()
