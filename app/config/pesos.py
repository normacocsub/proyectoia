import numpy as np
import os

ruta_archivo = os.path.join(os.path.dirname(__file__), '..', 'config', 'pesos.txt')

def crear_pesos(entradas, salidas):
    pesos = []
    if os.path.exists(ruta_archivo):
        pesos = np.loadtxt(ruta_archivo).reshape(entradas, salidas)
    else:
        pesos = np.random.uniform(-1, 1, size=(entradas, salidas)).reshape(entradas, salidas)
    return pesos
