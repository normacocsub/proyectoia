from app.routes.red import router as red_router
from fastapi import FastAPI, Request, Response, UploadFile, WebSocket, File
import websockets
from app.utils.red import simular_keras, entrenar_keras, cantidad_imagenes
from io import BytesIO
from PIL import Image
import base64
import json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
connected_clients = set()

# Configuración del CORS
origins = [
    "http://localhost:3000",  # Agrega aquí los orígenes permitidos
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




@app.middleware("http")
async def custom_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Custom-Header"] = "Custom header value"
    return response



@app.get("/")
async def root():
    return {"message": "La aplicacion esta funcionando!"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)

    try:
        while True:
            await simular_red(websocket)
    except websockets.exceptions.ConnectionClosed:
        connected_clients.remove(websocket)

@app.websocket("/wse")
async def websocket_entrenamiento(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            await entrenar_red(websocket)
    except websockets.exceptions.ConnectionClosed:
        connected_clients.remove(websocket)   

async def entrenar_red(websocket: WebSocket):
    while True:
        message = await websocket.receive()
        data = message["text"]
        json_data = json.loads(data)
        #error_response = await entrenar_perceptron(json_data['iterations'], json_data['error_maximo'], json_data['tasa_aprendizaje'], websocket)
        error_response, total_epoch = await entrenar_keras(json_data['iterations'], json_data['error_maximo'], json_data['tasa_aprendizaje'], websocket)
        await websocket.send_text(json.dumps({"lost_response": error_response, "finish": 1, "iteration": (total_epoch), "iterations": json_data['iterations']}))

async def simular_red(websocket: WebSocket):
    while True:
        message = await websocket.receive()
        data = message["text"]
        if data is not None:
            # Extraer el contenido base64 de los datos
            _, encoded_data = data.split(",")
            # Decodificar los datos base64
            decoded_data = base64.b64decode(encoded_data)
            # Abrir la imagen utilizando BytesIO
            image = Image.open(BytesIO(decoded_data))
            image = image.convert("RGB")
            # Procesar la imagen, realizar predicciones, etc.
            predictions = simular_keras(image, connected_clients)
            # Enviar las predicciones de vuelta al cliente WebSocket
            await websocket.send_text(predictions)

app.include_router(red_router, prefix="/red", tags=["red"])
