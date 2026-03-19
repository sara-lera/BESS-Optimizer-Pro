import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# -- CONFIGURACIÓN DE PÁGINA --
st.set_page_config(page_title="Eco-Optimizer BESS Dashboard", layout="wide", page_icon="⚡")

st.title("⚡ Eco-Optimizer BESS: Comparador de Modelos")
st.markdown("Analiza gráficamente el rendimiento de los modelos predictivos del precio de energía (15-min) día a día.")

@st.cache_data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data", "raw", "dataset_multivariante_completo.csv")
    if not os.path.exists(data_path):
        return None
    df = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
    return df.dropna()

def create_xgb_features(df):
    df_feat = df.copy()
    df_feat['hour'] = df_feat.index.hour
    df_feat['minute'] = df_feat.index.minute
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['lag_24h'] = df_feat['precio_mwh'].shift(96)
    return df_feat.dropna()

@st.cache_data(show_spinner=True)
def run_models(_df, days_test=3):
    """Entrena modelos rápidos y devuelve las predicciones de los últimos N días."""
    n_test = 96 * days_test
    
    # Baseline SARIMA
    train_s = _df.iloc[:-n_test]
    test_s = _df.iloc[-n_test:]
    
    model_sarima = SARIMAX(train_s['precio_mwh'], 
                           exog=train_s[['demanda_real']], 
                           order=(1, 1, 1))
    fit_sarima = model_sarima.fit(disp=False)
    preds_sarima = fit_sarima.predict(start=len(train_s), end=len(train_s)+len(test_s)-1, exog=test_s[['demanda_real']])
    preds_sarima.index = test_s.index
    
    # Random Forest
    df_rf = create_xgb_features(_df) # reutilizar features
    train_rf = df_rf.iloc[:-n_test]
    test_rf = df_rf.iloc[-n_test:]
    features = [c for c in train_rf.columns if c != 'precio_mwh']
    
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model_rf.fit(train_rf[features], train_rf['precio_mwh'])
    preds_rf = model_rf.predict(test_rf[features])
    preds_rf = pd.Series(preds_rf, index=test_rf.index)
    
    # Juntar todo
    results = pd.DataFrame({
        'Real': test_rf['precio_mwh'],
        'SARIMA': preds_sarima.reindex(test_rf.index).fillna(method='bfill'),
        'RandomForest': preds_rf
    })
    
    # XGBoost SOTA Optional
    if HAS_XGB:
        model_xgb = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
        model_xgb.fit(train_rf[features], train_rf['precio_mwh'])
        preds_xgb = model_xgb.predict(test_rf[features])
        results['XGBoost'] = pd.Series(preds_xgb, index=test_rf.index)
    
    return results

# Cargar Datos
df = load_data()

if df is None:
    st.error("No se encontró `dataset_multivariante_completo.csv`. ¡Asegúrate de haber ejecutado data_ingestion.py!")
    st.stop()

with st.spinner("Entrenando modelos y generando predicciones (esto puede tardar unos segundos la primera vez)..."):
    results = run_models(df, days_test=4)

# Métrica General
st.subheader("📊 Métricas Globales (Últimos 4 días)")
model_names = ['SARIMA', 'RandomForest']
if HAS_XGB:
    model_names.insert(1, 'XGBoost')

cols = st.columns(len(model_names))

for idx, m_name in enumerate(model_names):
    mae = mean_absolute_error(results['Real'], results[m_name])
    rmse = np.sqrt(mean_squared_error(results['Real'], results[m_name]))
    cols[idx].metric(label=f"🏆 {m_name}", value=f"MAE: €{mae:.2f}", delta=f"RMSE: €{rmse:.2f}", delta_color="inverse")

st.markdown("---")

# Filtro interactivo por Día
st.subheader("📅 Análisis Día a Día")
unique_days = np.unique(results.index.date)
selected_day = st.selectbox("Selecciona el día para ver al detalle:", unique_days)

# Filtrar el dataframe por el día seleccionado
mask = results.index.date == selected_day
day_data = results[mask]

# Gráfico interactivo con Plotly
fig = go.Figure()

# Linea valor Real
fig.add_trace(go.Scatter(x=day_data.index, y=day_data['Real'], mode='lines', 
                         name='Precio Real', line=dict(color='white', width=3)))

# Lineas predicciones
colors = {'SARIMA': '#ff9900', 'XGBoost': '#00ffcc', 'RandomForest': '#ff3366'}
for m_name in model_names:
    fig.add_trace(go.Scatter(x=day_data.index, y=day_data[m_name], mode='lines', 
                             name=m_name, line=dict(dash='dot', width=2, color=colors[m_name])))

fig.update_layout(
    title=f"Predicciones vs Real ({selected_day})",
    xaxis_title="Hora",
    yaxis_title="Precio (€/MWh)",
    hovermode="x unified",
    template="plotly_dark",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<style>
/* Ajustes visuales para hacerlo más premium */
.css-18e3th9 { padding-top: 2rem; }
h1 { color: #00ffcc; font-weight: 700; }
</style>
""", unsafe_allow_html=True)
