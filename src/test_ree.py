import requests
import pandas as pd

url_generacion = "https://apidatos.ree.es/es/datos/generacion/estructura-generacion"
params = {
    "start_date": "2026-01-01T00:00",
    "end_date": "2026-01-01T23:59",
    "time_trunc": "hour",
    "geo_trunc": "electric_system",
    "geo_limit": "peninsular",
    "geo_ids": "8741"
}
r = requests.get(url_generacion, params=params)
print("With hour:", r.status_code)

# Try removing geo params
params2 = {
    "start_date": "2026-01-01T00:00",
    "end_date": "2026-01-01T23:59",
    "time_trunc": "hour"
}
r2 = requests.get(url_generacion, params=params2)
print("No geo, hour:", r2.status_code)

# Try fetching from a different endpoint e.g. generacion/evolucion-renovable-no-renovable
url_ren = "https://apidatos.ree.es/es/datos/generacion/evolucion-renovable-no-renovable"
r3 = requests.get(url_ren, params=params)
print("Evolucion renovable hour:", r3.status_code)

