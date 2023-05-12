import os
import numpy as np

ruta_archivo = os.path.join(os.path.dirname(__file__), '..', 'config', 'umbrales.txt')

def crear_umbrales(salidas):
    umbrales = []
    if os.path.exists(ruta_archivo):
        umbrales = np.atleast_1d(np.loadtxt(ruta_archivo))
    else:
        umbrales = np.random.uniform(-1, 1, size=(salidas))
    return umbrales