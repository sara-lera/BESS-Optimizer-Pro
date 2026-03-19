import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def create_features(df):
    """Crea features temporales (SOTA para tabular time-series)"""
    df_feat = df.copy()
    df_feat['hour'] = df_feat.index.hour
    df_feat['minute'] = df_feat.index.minute
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['lag_24h'] = df_feat['precio_mwh'].shift(96) # 24h a 15min
    df_feat['lag_1w'] = df_feat['precio_mwh'].shift(96*7) # 1 semana
    return df_feat.dropna()

def main():
    print("--- Modelo XGBoost (SOTA Tabular) ---")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "raw", "dataset_multivariante_completo.csv")
    
    df = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
    df = create_features(df)
    
    n_test = 96 * 2 # 2 dias
    train = df.iloc[:-n_test]
    test = df.iloc[-n_test:]
    
    features = [c for c in train.columns if c != 'precio_mwh']
    
    X_train, y_train = train[features], train['precio_mwh']
    X_test, y_test = test[features], test['precio_mwh']
    
    # XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    print("Ajustando XGBoost...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    print("Prediciendo...")
    preds = model.predict(X_test)
    preds = pd.Series(preds, index=test.index)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"XGBoost - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(test.index, y_test, label='Real')
    plt.plot(test.index, preds, label='XGBoost Predict', linestyle='dashed')
    plt.title("Predicción XGBoost SOTA")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(script_dir, "xgboost_plot.png"))
    print("Guardado en xgboost_plot.png")

if __name__ == "__main__":
    main()
