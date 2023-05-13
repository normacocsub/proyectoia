import numpy as np
from PIL import Image
import shutil

def cargar_archivo(ruta_archivo):
    with open(ruta_archivo, 'r') as archivo:
        lines = archivo.readlines()
    lines = [l.strip().split(';') for l in lines]

    header = lines[0]
    data = lines[1:]
    header_vector = [str(x) for x in header[0:]]
    s1_index = header_vector.index('s1')

    # Convierte las filas restantes en una matriz
    matrix_entrada = [[float(x) for x in row[0:s1_index]] for row in data]
    matrix_entrada_np = np.array(matrix_entrada)
    num_filas, entradas_total = matrix_entrada_np.shape

    matrix_salida = [[float(x) for x in row[s1_index:]] for row in data]
    matrix_salida_np = np.array(matrix_salida)
    num_filas, salidas_total = matrix_salida_np.shape

    return matrix_entrada_np, matrix_salida_np, num_filas, entradas_total, salidas_total

def cargar_imagen(file):
    with open(file.filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # Abrir la imagen con Pillow
    imagen_color = Image.open(file.filename)

    imagen_reducida = imagen_color.resize((150, 150))

    # Convertir la imagen a escala de grises
    imagen_gris = imagen_reducida.convert('L')

    # Convertir la imagen en una matriz numpy
    matriz_imagen = np.array(imagen_gris)

    # Convertir la matriz en una matriz de 0 y 1 con un umbral de 128
    matriz_binaria = (matriz_imagen >= 128).astype(int)

    # Sumar las filas y columnas de la matriz binaria
    x_train_sum_filas = np.sum(matriz_binaria, axis=1)
    x_train_sum_columnas = np.sum(matriz_binaria, axis=0)


    # Concatenar los resultados para obtener un vector de 1x300
    x_train_sum = np.concatenate((x_train_sum_filas, x_train_sum_columnas), axis=0)
    x_train_sum = x_train_sum.reshape(1, -1)
    return x_train_sum