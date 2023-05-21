from fastapi import APIRouter, status, UploadFile, File
from fastapi.responses import JSONResponse
import os
from app.utils.file import cargar_archivo, cargar_imagen
from app.utils.red import train_red, iniciar_red, simular, entrenar_keras, simular_keras
from app.schemas import RedModel
from app.config import pesos, umbrales
from app.utils.constantes import find_object
import asyncio
from fastapi import WebSocket





router = APIRouter()


ruta_archivo = os.path.join(os.path.dirname(__file__), '..', 'config', 'banco_datos.txt')


@router.post("/", status_code=status.HTTP_201_CREATED)
def entrenar_red(red_model: RedModel):
    entradas, salidas, patrones, total_entradas, total_salidas = cargar_archivo(ruta_archivo)
    pesos_red, umbrales_red = iniciar_red(total_salidas, total_entradas)
    red_is_traint ,error_iteracion = train_red(entradas, salidas, patrones, total_salidas, total_entradas, red_model.iteraciones, red_model.tasa_aprendizaje, red_model.error_maximo, pesos_red, umbrales_red)
    return {"error_iteration":error_iteracion}, {"red_train":red_is_traint}

@router.post("/entrenar", status_code=status.HTTP_201_CREATED)
def entrenar_red_keras():
    error, error_validacion = entrenar_keras()
    return error, error_validacion


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




