import numpy as np
import os

ruta_archivo = os.path.join(os.path.dirname(__file__), '..', 'config', 'pesos.txt')
ruta_archivo_config = os.path.join(os.path.dirname(__file__), '..', 'config', 'config_red.txt')

def crear_pesos(entradas, salidas):
    pesos = []
    if os.path.exists(ruta_archivo) and os.path.exists(ruta_archivo_config):
        with open(ruta_archivo_config, 'r') as archivo:
            lines = archivo.readlines()
        lines = [l.strip().split(';') for l in lines]
        datos = lines[0]
        if datos[0] == "0":
            pesos = np.random.uniform(-1, 1, size=(entradas, salidas)).reshape(entradas, salidas)
            np.savetxt(ruta_archivo, pesos)
        else:
            pesos = np.loadtxt(ruta_archivo).reshape(entradas, salidas)
    else:
        pesos = np.random.uniform(-1, 1, size=(entradas, salidas)).reshape(entradas, salidas)
        np.savetxt(ruta_archivo, pesos)
    return pesos

def cargar_pesos():
    pesos = []
    if os.path.exists(ruta_archivo):
        pesos = np.loadtxt(ruta_archivo)
        if len(pesos.shape) == 1:
            filas = len(pesos)
            pesos = pesos.reshape((filas, 1))
    return pesos

def guardar_pesos(pesos):
    np.savetxt(ruta_archivo, pesos)

def reiniciar_pesos(entradas, salidas):
    pesos = np.random.uniform(-1, 1, size=(entradas, salidas)).reshape(entradas, salidas)
    np.savetxt(ruta_archivo, pesos)
    return pesos
