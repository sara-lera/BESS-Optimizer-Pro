import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import pulp
import datetime
import sys

# Configuraciones de pagina
st.set_page_config(page_title="Eco-Optimizer BESS Pro", layout="wide", page_icon="⚡")

# Importar funciones externas
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, 'src'))
from data_ingestion import fetch_mercado_trinidad

# Importar algoritmos ML 
from sklearn.ensemble import RandomForestRegressor
try:
    import xgboost as xgb
except ImportError:
    xgb = None

# --- ESTILOS ---
st.markdown("""
<style>
/* Streamlit Dark Theme tweaks */
h1, h2, h3 { color: #00ffcc; font-weight: 700; }
.stTabs [data-baseweb="tab-list"] { gap: 2rem; }
.stTabs [data-baseweb="tab"] { font-size: 1.25rem; font-weight: 600; }
.stButton>button { border-color: #00ffcc; color: #00ffcc; width: 100%; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Eco-Optimizer BESS: End-to-End Pipeline")

# Inicializar estados
if 'df_hist' not in st.session_state:
    st.session_state.df_hist = None
if 'df_preds' not in st.session_state:
    st.session_state.df_preds = None

# ---- TABS ----
tab1, tab2, tab3 = st.tabs(["📥 1. Datos Históricos", "🔮 2. Predicción a Futuro", "🔋 3. Optimizador BESS"])

# ==========================================
# TAB 1: DATOS HISTORICOS Y DESCARGA
# ==========================================
with tab1:
    st.header("Descarga e Inspección de Datos del Mercado")
    st.markdown("Extrae los datos (Precio, Generación, Demanda) directamente de la API de Red Eléctrica.")
    
    col_d1, col_d2, col_btn = st.columns([2, 2, 1])
    with col_d1:
        s_date = st.date_input("Fecha Inicio", pd.to_datetime("2026-01-01"))
    with col_d2:
        e_date = st.date_input("Fecha Fin", pd.to_datetime("2026-01-07"))
        
    with col_btn:
        st.write("")
        st.write("")
        if st.button("⬇️ Descargar Datos de REE"):
            with st.spinner("Descargando, fusionando e interpolando a 15 min..."):
                try:
                    df = fetch_mercado_trinidad(s_date.strftime("%Y-%m-%d"), e_date.strftime("%Y-%m-%d"))
                    
                    # Calcular Generación Total sumando todas las fuentes si está desglose
                    gen_cols = [c for c in df.columns if c not in ['precio_mwh', 'demanda']]
                    if len(gen_cols) > 0:
                        df['generacion_total'] = df[gen_cols].sum(axis=1)
                    else:
                        df['generacion_total'] = df['demanda'] # Fallback por si acaso
                        
                    st.session_state.df_hist = df
                    st.success(f"¡Datos importados con éxito! {len(df)} registros.")
                except Exception as e:
                    st.error(f"Error descargando datos: {str(e)}")
                    
    # Visualizacion si ya hay datos
    if st.session_state.df_hist is not None:
        df_plot = st.session_state.df_hist
        st.markdown("---")
        
        c1, c2 = st.columns([1, 4])
        with c1:
            st.write("🔧 Controles de Visualización")
            var_base = st.selectbox("Selecciona la métrica principal:", ['Precio (MWh)', 'Demanda', 'Generación'])
            
            gen_type = 'Total'
            if var_base == 'Generación':
                gen_type = st.radio("Desglose de Generación:", ['Total', 'Por Tecnología'])
                
        with c2:
            fig = go.Figure()
            
            # Determinar si mostrar puntos
            dias_totales = (df_plot.index.max() - df_plot.index.min()).days
            mode_plot = 'lines+markers' if dias_totales <= 2 else 'lines'
            
            if var_base == 'Precio (MWh)':
                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['precio_mwh'], mode=mode_plot, name='Precio', line=dict(color='#00ffcc', width=2)))
            elif var_base == 'Demanda':
                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['demanda'], mode=mode_plot, name='Demanda', line=dict(color='#ff9900', width=2)))
            elif var_base == 'Generación':
                if gen_type == 'Total':
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['generacion_total'], mode=mode_plot, name='Generación Total', line=dict(color='#ff3366', width=2)))
                else:
                    gcols = [c for c in df_plot.columns if c not in ['precio_mwh', 'demanda', 'generacion_total']]
                    for col in gcols:
                        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[col], mode='lines', stackgroup='one', name=col.replace('_', ' ').title()))
            
            fig.update_layout(
                title=f"Evolución Histórica: {var_base}", 
                template="plotly_dark", 
                height=500,
                xaxis=dict(rangeslider=dict(visible=True), type="date")
            )
            st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: PREDICCIONES
# ==========================================
with tab2:
    st.header("Predicciones de Precios y Sistema Eléctrico")
    if st.session_state.df_hist is None:
        st.info("⚠️ Debes descargar los datos históricos en la Tabla 1 antes de poder predecir.")
    else:
        df_train = st.session_state.df_hist
        st.write(f"Usando dataset cargado: del {df_train.index.min().strftime('%Y-%m-%d')} al {df_train.index.max().strftime('%Y-%m-%d')}.")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            horizon_options = {"1 Hora": 1, "4 Horas": 4, "1 Día": 24, "2 Días": 48, "1 Semana": 168}
            horizonte = st.selectbox("🔮 Horizonte a Futuro:", list(horizon_options.keys()))
        with c2:
            model_options = ["Random Forest SOTA"]
            if xgb is not None:
                model_options.insert(0, "XGBoost SOTA")
            selected_model = st.selectbox("🧠 Algoritmo de ML:", model_options)
        with c3:
            st.write("")
            st.write("")
            run_pred = st.button("🚀 Iniciar Predicción")
            
        if run_pred:
            with st.spinner(f"Entrenando modelo sobre TODO el dataset y prediciendo {horizonte}..."):
                steps_ahead = horizon_options[horizonte]
                
                # Crear variables predictivas de Tiempo (SOTA directo multi-step sin lag complejo)
                def make_features(idx):
                    return pd.DataFrame({
                        'hour': idx.hour,
                        'dayofweek': idx.dayofweek,
                    }, index=idx)
                
                X_train = make_features(df_train.index)
                
                # Para el test, extendemos el índice hacia el futuro
                last_time = df_train.index[-1]
                future_idx = pd.date_range(start=last_time + pd.Timedelta(hours=1), periods=steps_ahead, freq='1h')
                X_test = make_features(future_idx)
                
                # Model definition
                if "XGBoost" in selected_model:
                    model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42)
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    
                # Store predictions
                preds_df = pd.DataFrame(index=future_idx)
                
                # Train and predict variables required by User: generación_total, demanda, precio_mwh
                target_vars = ['generacion_total', 'demanda', 'precio_mwh']
                
                for var in target_vars:
                    if var in df_train.columns:
                        y_train = df_train[var]
                        model.fit(X_train, y_train)
                        preds_df[var] = model.predict(X_test)
                
                st.session_state.df_preds = preds_df
                st.success("¡Modelos entrenados y predicciones generadas!")
                
        # Mostrar gráfica ensamblada si hay predicción
        if st.session_state.df_preds is not None:
            df_preds = st.session_state.df_preds
            st.markdown("---")
            st.subheader("📊 Visualización Continua (Real + Predicción)")
            
            p_var = st.selectbox("Ver progresión de:", ['precio_mwh', 'demanda', 'generacion_total'], key='p_var')
            
            if p_var in df_train.columns and p_var in df_preds.columns:
                # Mostrar ultima parte del real para concatenar visual
                df_real_tail = df_train.iloc[-48:] # ver ultimos 2 dias maximo para contexto
                
                fig2 = go.Figure()
                
                # Plot modo dependiendo de la duración
                mode_plot2 = 'lines+markers' if (df_preds.index.max() - df_real_tail.index.min()).days <= 2 else 'lines'
                
                fig2.add_trace(go.Scatter(x=df_real_tail.index, y=df_real_tail[p_var], 
                                         mode=mode_plot2, name='Histórico Real', line=dict(color='white', width=2)))
                                         
                fig2.add_trace(go.Scatter(x=df_preds.index, y=df_preds[p_var], 
                                         mode=mode_plot2, name='Predicción Futura', line=dict(color='#ff3366', width=2, dash='dot')))
                                         
                fig2.add_vline(x=df_train.index[-1], line_width=2, line_dash="dash", line_color="yellow")
                fig2.update_layout(title=f"Dato Histórico Real empalmado con Predicción: {p_var.upper()}", template="plotly_dark", xaxis=dict(rangeslider=dict(visible=True), type="date"))
                st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# TAB 3: BESS OPTIMIZER
# ==========================================
with tab3:
    st.header("🔋 Optimización BESS (Arbitraje sobre Predicción)")
    if st.session_state.df_preds is None:
        st.info("⚠️ Genera antes un horizonte de predicción de precios en la Pestaña 2.")
    else:
        st.markdown("Utiliza el Algoritmo Lineal LP (`pulp`) sobre los *precios del mercado a futuro* predichos para calcular cuándo comprar/vender.")
        
        preds = st.session_state.df_preds
        
        c1, c2, c3 = st.columns(3)
        cap = c1.number_input("Capacidad BESS (kWh)", value=2000)
        power = c2.number_input("Potencia Máxima (kW)", value=1000)
        eff = c3.slider("Eficiencia Batería (Round-trip) %", min_value=50, max_value=100, value=90) / 100.0
        
        if st.button("⚖️ Calcular Despacho Óptimo BESS"):
            with st.spinner("Modelando el problema de Arbitraje..."):
                precios_kwh = preds['precio_mwh'].values / 1000.0
                n_steps = len(preds)
                dt = 1.0 # 1 hora
                
                prob = pulp.LpProblem("Dashboard_BESS", pulp.LpMaximize)
                indices = range(n_steps)
                
                # Variables
                p_c = pulp.LpVariable.dicts("P_charge", indices, 0, power)
                p_d = pulp.LpVariable.dicts("P_discharge", indices, 0, power)
                soc = pulp.LpVariable.dicts("SoC", indices, 0, cap)
                
                # Funcion objetivo
                beneficio = pulp.lpSum([precios_kwh[t] * p_d[t] * dt - precios_kwh[t] * p_c[t] * dt for t in indices])
                prob += beneficio
                
                # Restricciones
                initial_soc = 0.5 * cap
                for t in indices:
                    if t == 0:
                        prob += soc[t] == initial_soc + p_c[t]*eff*dt - (p_d[t]/eff)*dt
                    else:
                        prob += soc[t] == soc[t-1] + p_c[t]*eff*dt - (p_d[t]/eff)*dt
                        
                # Solve
                prob.solve()
                
                # Result Parse
                charge_vals = [p_c[t].varValue for t in indices]
                discharge_vals = [p_d[t].varValue for t in indices]
                soc_vals = [soc[t].varValue for t in indices]
                
                st.success(f"¡Optimización resuelta! (Estado: {pulp.LpStatus[prob.status]}) | Beneficio Estimado Horizonte: **{pulp.value(prob.objective):.2f} €**")
                
                # Plot final
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=preds.index, y=preds['precio_mwh'], mode='lines', name='Precio Predicho (EUR)', yaxis='y1', line=dict(color='#00ffcc')))
                fig3.add_trace(go.Bar(x=preds.index, y=discharge_vals, name='Descarga (Venta kW)', yaxis='y2', marker_color='green', opacity=0.6))
                fig3.add_trace(go.Bar(x=preds.index, y=[-c for c in charge_vals], name='Carga (Compra kW)', yaxis='y2', marker_color='red', opacity=0.6))
                fig3.add_trace(go.Scatter(x=preds.index, y=(np.array(soc_vals)/cap)*100, mode='lines', name='SoC (%)', yaxis='y3', line=dict(color='#ff3366', dash='dot')))
                
                fig3.update_layout(
                    title="Planificador Óptimo de Compras y Ventas del BESS",
                    template="plotly_dark",
                    hovermode="x unified",
                    yaxis=dict(title="Precio (EUR/MWh)", side='left', showgrid=False),
                    yaxis2=dict(title="Potencia Batería (kW)", side='right', overlaying='y', showgrid=False),
                    yaxis3=dict(title="Estado de Carga - SoC (%)", side='right', overlaying='y', position=0.95, range=[0, 105], showgrid=False),
                    legend=dict(orientation="h", x=0, y=1.1)
                )
                
                st.plotly_chart(fig3, use_container_width=True)
