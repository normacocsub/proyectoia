from app.routes.red import router as red_router
from fastapi import FastAPI, Request, Response, UploadFile, WebSocket, File
import websockets
from app.utils.red import simular_keras
from io import BytesIO
from PIL import Image
import base64

app = FastAPI()
connected_clients = set()




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

async def simular_red(websocket: WebSocket):
    while True:
        message = await websocket.receive()
        data = message["text"]

        # Extraer el contenido base64 de los datos
        _, encoded_data = data.split(",")
        # Decodificar los datos base64
        decoded_data = base64.b64decode(encoded_data)

        # Abrir la imagen utilizando BytesIO
        image = Image.open(BytesIO(decoded_data))

        # Convertir la imagen en un objeto bytes
        image_bytes = image.tobytes()

        # Procesar la imagen, realizar predicciones, etc.
        predictions = simular_keras(image, connected_clients)

        

        # Enviar las predicciones de vuelta al cliente WebSocket
        await websocket.send_text(predictions)

app.include_router(red_router, prefix="/red", tags=["red"])
