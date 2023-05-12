from app.routes.red import router as red_router
from fastapi import FastAPI, Request, Response



app = FastAPI()




@app.middleware("http")
async def custom_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Custom-Header"] = "Custom header value"
    return response


@app.get("/")
async def root():
    return {"message": "La aplicacion esta funcionando!"}


app.include_router(red_router, prefix="/red", tags=["red"])
