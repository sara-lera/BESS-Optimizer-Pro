import requests

session = requests.Session()
urls = {
    "gen_tr": "https://apidatos.ree.es/es/datos/generacion/estructura-tiempo-real",
    "dem_tr": "https://apidatos.ree.es/es/datos/demanda/tiempo-real"
}
params = {
    "start_date": "2026-01-01T00:00",
    "end_date": "2026-01-01T23:59",
    "time_trunc": "hour",  # 'hour' para que lo agregue si lo soporta
    "geo_trunc": "electric_system",
    "geo_limit": "peninsular",
    "geo_ids": "8741"
}

for name, url in urls.items():
    r = session.get(url, params=params)
    print(f"Testing {name}: Status {r.status_code}")
    if r.status_code == 200:
        data = r.json().get('included', [])
        if data:
            vals = data[0]['attributes']['values']
            if len(vals) > 0:
                print(f"  First: {vals[0]['datetime']}, count: {len(vals)}")
