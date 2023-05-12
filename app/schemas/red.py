from pydantic import BaseModel, conint, confloat

class RedModel(BaseModel):
    iteraciones: conint(gt=0)
    tasa_aprendizaje: confloat(gt=0, lt=1)
    error_maximo: confloat(gt=0)