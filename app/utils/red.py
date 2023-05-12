from app.config import pesos, umbrales

def train_red(datos_entrada, datos_salida,patrones, total_salidas, total_entradas, iteraciones, rata_aprendizaje, error_maximo):
    pesos_red =  pesos.crear_pesos(total_entradas, total_salidas)
    umbrales_red = umbrales.crear_umbrales(total_salidas)
    return ""
