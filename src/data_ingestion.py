import requests
import pandas as pd
import time
from datetime import datetime
import os

def fetch_mercado_trinidad(start_date: str, end_date: str, progress_callback=None) -> pd.DataFrame:
    # Los 3 endpoints clave
    url_precios = "https://apidatos.ree.es/es/datos/mercados/precios-mercados-tiempo-real"
    url_generacion = "https://apidatos.ree.es/es/datos/generacion/estructura-generacion"
    url_demanda = "https://apidatos.ree.es/es/datos/demanda/evolucion"
    
    fechas = pd.date_range(start=start_date, end=end_date, freq='D')
    
    datos_precios = []
    dict_generacion = {}
    dict_demanda = {}
    
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Accept-Language": "es-ES,es;q=0.9"
    })
    
    print(f"Arrancando extracción TOTAL (Precio -> Generación -> Demanda) desde {start_date} a {end_date}...")
    
    for i, fecha in enumerate(fechas):
        if progress_callback:
            progress_callback(int((i / len(fechas)) * 90), f"Descargando día {i+1} de {len(fechas)} ({fecha.strftime('%Y-%m-%d')})...")
            
        dia_str = fecha.strftime('%Y-%m-%d')
        print(f"-> Procesando día: {dia_str}")
        
        params_precio = {
            "start_date": f"{dia_str}T00:00",
            "end_date": f"{dia_str}T23:59",
            "time_trunc": "hour"
        }
        
        # Parámetros geográficos obligatorios para Generación y Demanda
        params_geo = params_precio.copy()
        params_geo.update({
            "geo_trunc": "electric_system",
            "geo_limit": "peninsular",
            "geo_ids": "8741"
        })
        
        # 1. PRECIOS
        res_precio = session.get(url_precios, params=params_precio)
        if res_precio.status_code == 200:
            try:
                df_p = pd.DataFrame(res_precio.json()['included'][0]['attributes']['values'])
                df_p = df_p[['datetime', 'value']].rename(columns={'value': 'precio_mwh'})
                df_p['datetime'] = pd.to_datetime(df_p['datetime'], utc=True).dt.tz_convert('Europe/Madrid').dt.tz_localize(None)
                df_p.set_index('datetime', inplace=True)
                datos_precios.append(df_p)
            except Exception as e: 
                print(f"  [!] Error parseando precios: {e}")
        else:
            print(f"  [!] Fallo API Precios: {res_precio.status_code}")
                
        time.sleep(1) 
        
        # 2. GENERACIÓN
        params_gen = params_geo.copy()
        # OBLIGATORIO: REE API devuelve error 500 para estructura-generacion con time_trunc=hour
        # Por lo tanto, forzamos time_trunc='day' y luego en el resample dividimos entre 24 para aplanarlo.
        params_gen["time_trunc"] = "day"
        res_gen = session.get(url_generacion, params=params_gen)
        if res_gen.status_code == 200:
            try:
                for tech in res_gen.json()['included']:
                    tipo = tech['type']
                    col_name = tipo.lower().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
                    
                    df_g = pd.DataFrame(tech['attributes']['values'])[['datetime', 'value']]
                    df_g = df_g.rename(columns={'value': col_name})
                    df_g['datetime'] = pd.to_datetime(df_g['datetime'], utc=True).dt.tz_convert('Europe/Madrid').dt.tz_localize(None)
                    df_g.set_index('datetime', inplace=True)
                    
                    if col_name not in dict_generacion:
                        dict_generacion[col_name] = []
                    dict_generacion[col_name].append(df_g)
            except Exception as e:
                print(f"  [!] Error parseando generación: {e}")
        else:
            print(f"  [!] Fallo API Generación: {res_gen.status_code}")
                
        time.sleep(1) 
        
        # 3. DEMANDA
        res_dem = session.get(url_demanda, params=params_geo)
        if res_dem.status_code == 200:
            try:
                for dem in res_dem.json()['included']:
                    tipo = dem['type']
                    # Limpiamos nombre (suele venir "Demanda real", "Demanda prevista", etc.)
                    col_name = tipo.lower().replace(' ', '_')
                    
                    df_d = pd.DataFrame(dem['attributes']['values'])[['datetime', 'value']]
                    df_d = df_d.rename(columns={'value': col_name})
                    df_d['datetime'] = pd.to_datetime(df_d['datetime'], utc=True).dt.tz_convert('Europe/Madrid').dt.tz_localize(None)
                    df_d.set_index('datetime', inplace=True)
                    
                    if col_name not in dict_demanda:
                        dict_demanda[col_name] = []
                    dict_demanda[col_name].append(df_d)
            except Exception as e:
                print(f"  [!] Error parseando demanda: {e}")
        else:
            print(f"  [!] Fallo API Demanda: {res_dem.status_code}")
            
        time.sleep(1)
        
    if not datos_precios:
        raise Exception("Fallo total: No se ha podido descargar el core (precios).")
        
    if progress_callback:
        progress_callback(90, "Ensamblando datos y cruzando tablas...")
        
    # Ensamblaje del monstruo
    df_master = pd.concat(datos_precios)
    df_master = df_master[~df_master.index.duplicated(keep='first')]
    
    print("\nFusionando Generación...")
    for tech_name, lista_dfs in dict_generacion.items():
        if lista_dfs:
            df_tech = pd.concat(lista_dfs)
            df_tech = df_tech[~df_tech.index.duplicated(keep='first')]
            df_master = df_master.join(df_tech, how='outer')
            
    print("Fusionando Demanda...")
    for dem_name, lista_dfs in dict_demanda.items():
        if lista_dfs:
            df_dem = pd.concat(lista_dfs)
            df_dem = df_dem[~df_dem.index.duplicated(keep='first')]
            df_master = df_master.join(df_dem, how='outer')
    
    print("Alineando a frecuencia Horaria Continua...")
    # Precios (de 15min -> Promedio 1H)
    pcols = ['precio_mwh']
    df_p = df_master[pcols].resample('1h').mean()
    
    # Demanda (ya viene en 1H)
    dcols = list(dict_demanda.keys())
    df_d = df_master[dcols].resample('1h').first()
    
    # Generacion (Viene 1 vez al día a las 00:00 como Total de Energía)
    # 1. Agrupamos por día para asegurar 1 solo registro (Total) por día válido
    gcols = [c for c in df_master.columns if c not in pcols + dcols]
    df_g = df_master[gcols].dropna(how='all').resample('1D').first()
    
    # 2. Reindexamos a horario extendiendo el total diario a todas las horas de ese día
    df_g_hourly = df_g.reindex(df_p.index).ffill()
    
    # 3. Calculamos la demanda total diaria para generar un "perfil de pesos horarios" exacto
    # Esto distribuye la generación matemáticamente siguiendo la curva de carga real, evitando el flat-line
    df_d_daily_sum = df_d.resample('1D').sum().reindex(df_p.index).ffill()
    
    col_demanda = 'demanda' if 'demanda' in df_d.columns else dcols[0] if dcols else None
    
    if col_demanda:
        # Peso de cada hora = (demanda de la hora) / (demanda total del día)
        pesos_horarios = df_d[col_demanda] / df_d_daily_sum[col_demanda]
        pesos_horarios = pesos_horarios.fillna(1.0/24.0) # Seguro
        
        for c in gcols:
            df_g_hourly[c] = df_g_hourly[c] * pesos_horarios
    else:
        df_g_hourly = df_g_hourly / 24.0 # Fallback extremo
    
    df_final = df_p.join(df_d, how='outer').join(df_g_hourly, how='outer').fillna(0)
    
    if progress_callback:
        progress_callback(100, "¡Proceso de datos finalizado!")
        
    return df_final

if __name__ == "__main__":
    # Descargamos los primeros 10 días de 2026 (intervalos de 15 min nativos para precios)
    df_final = fetch_mercado_trinidad("2026-01-01", "2026-01-10") 
    
    print("\nDATASET MULTIVARIANTE CREADO:")
    print(f"Total columnas ({len(df_final.columns)}): {list(df_final.columns)}")
    print(df_final.head(4))
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(script_dir, "..", "data", "raw")
    os.makedirs(target_dir, exist_ok=True) 
    
    file_path = os.path.normpath(os.path.join(target_dir, "dataset_multivariante_completo.csv"))
    df_final.to_csv(file_path)
    
    print(f"\nDatos guardados en: {file_path}")