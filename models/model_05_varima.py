import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

def forecast_varima(df_full, target_var, steps_ahead):
    """
    Robust Auto-VARIMA with proper training/testing split.
    """
    var_cols = ['generacion_total', 'demanda', 'precio_mwh']
    # Use only training data for orders detection and fitting
    df_train = df_full.iloc[:-steps_ahead][var_cols].copy()
    
    if len(df_train) < 50: # Stability check
        return np.full(steps_ahead, df_train[target_var].iloc[-1]), 0, 0
    
    best_d = 0
    try:
        test_res = adfuller(df_train[target_var].dropna())
        if test_res[1] > 0.05:
            best_d = 1
            work_data = df_train.diff().dropna()
            if not work_data.empty:
                test_res_2 = adfuller(work_data[target_var])
                if test_res_2[1] > 0.05:
                    best_d = 2
    except:
        best_d = 1

    if best_d == 0:
        diff_train = df_train.copy()
    elif best_d == 1:
        diff_train = df_train.diff().dropna()
    else:
        diff_train = df_train.diff().diff().dropna()

    try:
        var_model = VAR(diff_train)
        max_lag = min(24, len(diff_train) // 10) # More conservative lag
        var_fit = var_model.fit(maxlags=max_lag, ic='aic')
        best_p = var_fit.k_ar

        forecast_diff = var_fit.forecast(diff_train.values[-best_p:] if best_p > 0 else diff_train.values[-1:], steps=steps_ahead)

        if best_d == 0:
            forecast_levels = forecast_diff
        elif best_d == 1:
            last_real = df_train.iloc[-1].values
            forecast_levels = np.cumsum(forecast_diff, axis=0) + last_real
        else:
            last_real = df_train.iloc[-1].values
            last_diff = df_train.diff().iloc[-1].values
            first_diff = np.cumsum(forecast_diff, axis=0) + last_diff
            forecast_levels = np.cumsum(first_diff, axis=0) + last_real

        idx_var = var_cols.index(target_var)
        return forecast_levels[:, idx_var], best_p, best_d
    except:
        # Final fallback to Naive if VAR fails
        return np.full(steps_ahead, df_train[target_var].iloc[-1]), 0, best_d
