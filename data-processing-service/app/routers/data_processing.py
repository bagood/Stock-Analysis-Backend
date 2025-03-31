from fastapi import APIRouter

router = APIRouter(
    prefix="/data-processing/api/v1",
    tags=["data-processing"]
)

@router.get("/")
async def root():
    return {"message": "Welcome to the routers of the Data-Processing Services"}