from app.config import pesos, umbrales
import os

ruta_archivo_config = os.path.join(os.path.dirname(__file__), '..', 'config', 'config_red.txt')

def iniciar_red(total_salidas, total_entradas):
    pesos_red =  pesos.crear_pesos(total_entradas, total_salidas)
    umbrales_red = umbrales.crear_umbrales(total_salidas)
    return pesos_red, umbrales_red

def train_red(datos_entrada, datos_salida, patrones, total_salidas, total_entradas, iteraciones, rata_aprendizaje, error_maximo, pesos_red, umbrales_red):
    error_iteration = []
    red_is_train = False
    for iteration in range(iteraciones):
        errores_permitidos = 0
        #iteramos cada patron
        for patron in range(patrones):
            salidas_neuronas = []
            errores = []
            error_permitido = 0
            #iteracion de las salidas
            for i in range(total_salidas):
                suma_entrada_pesos = 0
                for j in range(total_entradas):
                    suma_entrada_pesos += ((datos_entrada[patron][j] * pesos_red[j][i]))
                salida_neurona = suma_entrada_pesos - umbrales_red[i]
                if salida_neurona > 0:
                    salida_neurona = 1
                else:
                    salida_neurona = 0
                salidas_neuronas.append(salida_neurona)
                #calculamos el error 
                error = datos_salida[patron][i] - salidas_neuronas[i]
                errores.append(error)
                error_permitido += error
            # ajustamos umbrales y pesos
            for i in range(total_salidas):
                for j in range(total_entradas):
                    pesos_red[j][i] = pesos_red[j][i] + (rata_aprendizaje * errores[i] * datos_entrada[patron][j])
                umbrales_red[i] = umbrales_red[i] + (rata_aprendizaje * errores[i] * datos_salida[patron][i])
            #sumamos los errores permitidos por cada patron de la iteracion
            errores_permitidos += (abs(error_permitido) / total_salidas)
        #validamos el si el error permitido es menor al error maximo para ver si concluimos el entrenamiento.
        error_iteracion = (errores_permitidos / patrones)
        error_iteration.append(error_iteracion)
        if (error_iteracion <= error_maximo):
            #guardamos los umbrales y pesos en una matriz  
            pesos.guardar_pesos(pesos_red)
            umbrales.guardar_umbrales(umbrales_red)
            red_is_train = True
            with open(ruta_archivo_config, 'w') as f:
                f.write('1;1')
            break
    return red_is_train, error_iteration

def simular(entradas, pesos_red, umbrales_red, total_patrones, total_salidas, total_entradas):
    salidas = []
    for patron in range(total_patrones):
        salidas_neuronas = []
        for i in range(total_salidas):
            suma_entrada_pesos = 0
            # iteramos cada elemento del patron
            for j in range(total_entradas):
                suma_entrada_pesos += ((entradas[patron][j] * pesos_red[j][i]))
            salida_neurona = suma_entrada_pesos - umbrales_red[i]
            if salida_neurona > 0:
                salida_neurona = 1
            else:
                salida_neurona = 0
            salidas_neuronas.append(salida_neurona)
        salida =  " ".join(str(x) for x in salidas_neuronas)
        salidas.append(salida)
    return salidas

