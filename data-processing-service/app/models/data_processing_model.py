from typing import List
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

class QueryParamsDownloadStockPrice(BaseModel):
    emiten: str = Field(strict=True, max_length=4)
    country: str = Field(default='id', max_length=2)
    start_date: str = Field(default=(datetime.today() - timedelta(days=90)).strftime('%Y-%m-%d'), max_length=10)
    end_date: str = Field(default=datetime.today().strftime('%Y-%m-%d'), max_length=10)

class ResponseDownloadStockPrice(BaseModel):
    date: datetime = Field(alias='Date')
    open: float = Field(alias='Open')
    high: float = Field(alias='High')
    low: float = Field(alias='Low')
    close: float = Field(alias='Close')
    volume: float = Field(alias='Volume')
    dividends: float = Field(alias='Dividends')
    stock_splits: float = Field(alias='Stock Splits')

class ResponseListDownloadStockPrice(BaseModel):
    stock_price: List[ResponseDownloadStockPrice]