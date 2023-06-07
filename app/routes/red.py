from fastapi import APIRouter, status, UploadFile, File
from fastapi.responses import JSONResponse
import os
from app.utils.file import cargar_archivo, cargar_imagen
from app.utils.red import iniciar_red, simular, entrenar_keras, simular_keras, cantidad_imagenes
from app.schemas import RedModel
from app.config import pesos, umbrales
from app.utils.constantes import find_object
import asyncio
from fastapi import WebSocket





router = APIRouter()


ruta_archivo = os.path.join(os.path.dirname(__file__), '..', 'config', 'banco_datos.txt')



@router.get("/imagenes", status_code=status.HTTP_201_CREATED)
def entrenar_red_keras():
    respuesta = cantidad_imagenes()
    return respuesta


# @router.post("/simular", status_code=status.HTTP_201_CREATED)
# async def simular_red(file: UploadFile = File(...)):
#     contents = await file.read()
#     return simular_keras(contents)






# @router.post("/simular", status_code=status.HTTP_201_CREATED)
# def simular_red(file: UploadFile = File(...)):
#     matriz_entradas = cargar_imagen(file)
#     entradas, salidas, patrones, total_entradas, total_salidas = cargar_archivo(ruta_archivo)
#     pesos_red = pesos.cargar_pesos()
#     umbrales_red = umbrales.cargar_umbrales()
#     salidas = simular(matriz_entradas, pesos_red, umbrales_red, len(matriz_entradas), total_salidas, total_entradas)
#     os.remove(file.filename)
#     return {"salida_binaria": salidas, "objeto": find_object(next(iter(salidas)))}




