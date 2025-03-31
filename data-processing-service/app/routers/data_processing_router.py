from ..models import data_processing_model
from ..services import data_processing_service

from datetime import datetime
from typing import Annotated
from fastapi import APIRouter, Query

router = APIRouter(
    prefix="/data-processing/api/v1",
    tags=["data-processing"]
)

@router.get("/download")
async def download_stock_price(query_params: Annotated[data_processing_model.QueryParamsDownloadStockPrice, Query()]) -> data_processing_model.ResponseListDownloadStockPrice:
    emiten_code = f'{query_params.emiten}.{query_params.country}'
    start_date = datetime.strptime(query_params.start_date, '%Y-%m-%d').date()
    end_date = datetime.strptime(query_params.end_date, '%Y-%m-%d').date()
    response = data_processing_service.download_stock_price(emiten_code, start_date, end_date)
    
    return response