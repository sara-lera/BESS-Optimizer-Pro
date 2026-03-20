import pandas as pd
import numpy as np
import xgboost as xgb

def create_features(df, target_var):
    df_feat = df.copy()
    df_feat['hour'] = df_feat.index.hour
    df_feat['dayofweek'] = df_feat.index.dayofweek
    n = len(df)
    if n > 96:
        df_feat['lag_24h'] = df_feat[target_var].shift(96)
    if n > 96*7:
        df_feat['lag_1w'] = df_feat[target_var].shift(96*7)
    return df_feat

def forecast_xgboost(df_full, target_var, steps_ahead):
    df_feat = create_features(df_full, target_var)
    train = df_feat.iloc[:-steps_ahead].dropna()
    test = df_feat.iloc[-steps_ahead:]
    
    if train.empty:
        return np.full(steps_ahead, df_full[target_var].iloc[-steps_ahead-1])
        
    features = [c for c in train.columns if c != target_var]
    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(train[features], train[target_var])
    return model.predict(test[features])
