import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import requests
from io import StringIO
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
load_dotenv()


class ExtractData:
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.base_url = 'https://www.alphavantage.co/query'
        self.api_key = os.environ['KEY']
    def query(self):
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': self.symbol,
            'apikey': self.api_key,
            'outputsize': 'full',
            'datatype': 'csv'
        }
        response = requests.get(self.base_url, params=params)
        csv_data = response.text

        data = pd.read_csv(StringIO(csv_data))

        data['timestamp'] = pd.to_datetime(data['timestamp'])  # Convert timestamp to datetime
        filtered_data = data[(data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)]

        self.stock_df = filtered_data

        return filtered_data


if __name__ == "__main__":
    symbols = 'AAPL'
    start_date = '2019-01-01'
    end_date = '2022-06-01'

    extractor = ExtractData(symbols, start_date, end_date)
    df = extractor.query()
    df = df.set_index('timestamp')

