from fastapi import FastAPI

from .routers import data_processing

app = FastAPI()

app.include_router(data_processing.router)

@app.get("/")
async def root():
    return {"message": "Welcome to the main file for the Data-Processing Services"}