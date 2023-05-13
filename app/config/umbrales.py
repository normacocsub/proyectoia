import os
import numpy as np

ruta_archivo = os.path.join(os.path.dirname(__file__), '..', 'config', 'umbrales.txt')
ruta_archivo_config = os.path.join(os.path.dirname(__file__), '..', 'config', 'config_red.txt')

def crear_umbrales(salidas):
    umbrales = []
    if os.path.exists(ruta_archivo) and os.path.exists(ruta_archivo_config):
        with open(ruta_archivo_config, 'r') as archivo:
            lines = archivo.readlines()
        lines = [l.strip().split(';') for l in lines]
        datos = lines[0]
        if datos[0] == "0":
            umbrales = np.random.uniform(-1, 1, size=(salidas))
            np.savetxt(ruta_archivo, umbrales)
        else:
            umbrales = np.atleast_1d(np.loadtxt(ruta_archivo))
    else:
        umbrales = np.random.uniform(-1, 1, size=(salidas))
        np.savetxt(ruta_archivo, umbrales)
    return umbrales

def guardar_umbrales(umbrales):
    np.savetxt(ruta_archivo, umbrales)

def cargar_umbrales():
    umbrales = []
    if os.path.exists(ruta_archivo):
        umbrales = np.atleast_1d(np.loadtxt(ruta_archivo))
    return umbrales

def reiniciar_umbrales(salidas):
    umbrales = np.random.uniform(-1, 1, size=(salidas))
    np.savetxt(ruta_archivo, umbrales)
    return umbrales