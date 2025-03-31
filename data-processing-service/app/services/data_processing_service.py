import json
import yfinance as yf
from datetime import date

from ..models import data_processing_model

def download_stock_price(emiten_code: str, start_date: date, end_date: date) -> data_processing_model.ResponseListDownloadStockPrice:
    num_days = (end_date - start_date).days
    period = f'{num_days}d'
    data = yf.Ticker(emiten_code) \
            .history(period=period) \
            .to_json(orient='table')
    json_data = json.loads(data)
    
    return data_processing_model.ResponseListDownloadStockPrice(stock_price=json_data['data'])