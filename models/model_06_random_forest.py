import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def create_features(df, target_var):
    df_feat = df.copy()
    df_feat['hour'] = df_feat.index.hour
    df_feat['dayofweek'] = df_feat.index.dayofweek
    # Robust Lags: only add if enough history exists
    n = len(df)
    if n > 96:
        df_feat['lag_24h'] = df_feat[target_var].shift(96)
    if n > 96*7:
        df_feat['lag_1w'] = df_feat[target_var].shift(96*7)
    return df_feat

def forecast_random_forest(df_full, target_var, steps_ahead):
    df_feat = create_features(df_full, target_var)
    train = df_feat.iloc[:-steps_ahead].dropna()
    test = df_feat.iloc[-steps_ahead:]
    
    if train.empty:
        # Fallback if dropna cleared everything (too short dataset)
        return np.full(steps_ahead, df_full[target_var].iloc[-steps_ahead-1])
        
    features = [c for c in train.columns if c != target_var]
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(train[features], train[target_var])
    return model.predict(test[features])
