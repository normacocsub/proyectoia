from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
import os
from app.utils.file import cargar_archivo
from app.utils.red import train_red

from app.schemas import RedModel

router = APIRouter()


ruta_archivo = os.path.join(os.path.dirname(__file__), '..', 'config', 'banco_datos.txt')


@router.post("/", status_code=status.HTTP_201_CREATED)
def entrenar_red(red_model: RedModel):
    entradas, salidas, patrones, total_entradas, total_salidas = cargar_archivo(ruta_archivo)
    train_red(entradas, salidas, patrones, total_salidas, total_entradas, red_model.iteraciones, red_model.tasa_aprendizaje, red_model.error_maximo)
    return ""


