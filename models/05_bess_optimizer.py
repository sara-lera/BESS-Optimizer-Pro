import pandas as pd
import numpy as np
import os
import pulp
import matplotlib.pyplot as plt

def main():
    print("--- Optimizador BESS (Battery Energy Storage System) ---")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Cargamos las predicciones del modelo o simplemente usamos la realidad para simular el optimizador perfecto
    # Lo ideal en un pipeline BESS es usar las PREDICCIONES de precio_mwh de XGBoost/SARIMA
    # Para el MVP, cargaremos las predicciones si existen, o usamos el dataset test real
    data_path = os.path.join(script_dir, "..", "data", "raw", "dataset_multivariante_completo.csv")
    df = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
    
    # Cogemos las ultimas 24 horas (96 periodos de 15 mins) para planificar
    n_steps = 96
    df_opt = df.iloc[-n_steps:].copy()
    
    # Precios en EUR/MWh. Lo pasamos a EUR/kWh
    precios_kwh = df_opt['precio_mwh'].values / 1000.0
    
    # Parámetros del BESS (Sistema comercial tipico)
    CAPACITY_KWH = 2000.0   # Capacidad total: 2 MWh
    MAX_POWER_KW = 1000.0   # Potencia maxima de inversion: 1 MW
    EFFICIENCY = 0.90       # Eficiencia round-trip (descarga*carga ~ 90%)
    INITIAL_SOC = 0.5       # Estado de carga inicial (50%)
    dt = 0.25               # 15 mins = 0.25 horas
    
    # Creamos el problema de optimización Linear Programming
    # Objetivo: Maximizar beneficio (Ingresos por descarga - Coste de carga)
    prob = pulp.LpProblem("BESS_Optimization", pulp.LpMaximize)
    
    # Variables de decision para cada timestep (0 a 95)
    # p_charge: potencia cargada en KW
    # p_discharge: potencia descargada en KW
    # soc: estado de carga en kWh
    indices = range(n_steps)
    p_charge = pulp.LpVariable.dicts("P_charge", indices, lowBound=0, upBound=MAX_POWER_KW)
    p_discharge = pulp.LpVariable.dicts("P_discharge", indices, lowBound=0, upBound=MAX_POWER_KW)
    soc = pulp.LpVariable.dicts("SoC", indices, lowBound=0, upBound=CAPACITY_KWH)
    
    # Funcion Objetivo
    beneficio = pulp.lpSum(
        [precios_kwh[t] * p_discharge[t] * dt - precios_kwh[t] * p_charge[t] * dt for t in indices]
    )
    prob += beneficio
    
    # Restricciones
    for t in indices:
        # Dinámica del BESS (Ecuacion de la bateria)
        if t == 0:
            prob += soc[t] == INITIAL_SOC * CAPACITY_KWH + p_charge[t] * EFFICIENCY * dt - (p_discharge[t] / EFFICIENCY) * dt
        else:
            prob += soc[t] == soc[t-1] + p_charge[t] * EFFICIENCY * dt - (p_discharge[t] / EFFICIENCY) * dt
            
    # Resolvemos
    print("Calculando perfil óptimo (Arbitraje)...")
    prob.solve()
    print("Status:", pulp.LpStatus[prob.status])
    print(f"Beneficio Esperado Diario: {pulp.value(prob.objective):.2f} Euros")
    
    # Recoger resultados
    charge_vals = [p_charge[t].varValue for t in indices]
    discharge_vals = [p_discharge[t].varValue for t in indices]
    soc_vals = [soc[t].varValue for t in indices]
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.plot(df_opt.index, df_opt['precio_mwh'], color='black', label='Precio (EUR/MWh)', linewidth=2)
    ax1.set_ylabel('Precio (EUR/MWh)')
    ax1.set_xlabel('Tiempo')
    
    ax2 = ax1.twinx()
    ax2.bar(df_opt.index, discharge_vals, width=0.01, color='green', alpha=0.5, label='Descarga (Venta)')
    ax2.bar(df_opt.index, [-c for c in charge_vals], width=0.01, color='red', alpha=0.5, label='Carga (Compra)')
    ax2.plot(df_opt.index, np.array(soc_vals)/CAPACITY_KWH*100, color='blue', linestyle='--', label='SoC (%)')
    ax2.set_ylabel('Potencia (kW) / SoC (%)')
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    
    plt.title("Optimización de Despacho BESS - Arbitraje Perfecto de 24h")
    plt.grid(True)
    plt.tight_layout()
    out_img = os.path.join(script_dir, "bess_optimization.png")
    plt.savefig(out_img)
    plt.close()
    
    print(f"Gráfico de operación guardado en {out_img}")

if __name__ == "__main__":
    main()
