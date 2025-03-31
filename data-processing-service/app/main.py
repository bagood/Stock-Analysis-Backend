from fastapi import FastAPI

from .routers import data_processing_router

app = FastAPI()

app.include_router(data_processing_router.router)

@app.get("/")
async def root():
    return {"message": "Welcome to The Data-Processing Backend Service"}