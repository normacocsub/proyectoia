import numpy as np
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