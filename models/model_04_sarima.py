from statsmodels.tsa.statespace.sarimax import SARIMAX

def forecast_sarima(series, steps_ahead):
    try:
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
        fit_model = model.fit(disp=False)
        return fit_model.forecast(steps_ahead).values
    except:
        model = SARIMAX(series, order=(1, 1, 1))
        fit_model = model.fit(disp=False)
        return fit_model.forecast(steps_ahead).values
