import json
import os

ruta_archivo = os.path.join(os.path.dirname(__file__), '..', 'config', 'datos.json')

def find_object(value):
    with open(ruta_archivo, 'r') as f:
        data = json.load(f)
    print(data, value)
    for key, val in data.items():
        if val == value:
            return key
    return None

    