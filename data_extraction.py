import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import requests
from io import StringIO
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
load_dotenv()

class ExtractData:

    def __init__(self,symbols,start_date,end_date):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.base_url =  'https://www.alphavantage.co/query'
        self.api_key = os.environ['KEY']



    def query(self):
        stocks_df = pd.DataFrame({})

        for s in self.symbols:
            filtered_data = self.get_symbol(s)
            stocks_df[s] = filtered_data['close']

        stocks_df['timestamp'] = filtered_data['timestamp']
        self.stocks_df = stocks_df[['timestamp', 'T', 'VZ']]

    def plot(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.stocks_df['timestamp'], self.stocks_df['T'], label='AT&T')
        plt.plot(self.stocks_df['timestamp'], self.stocks_df['VZ'], label='Verizon')
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.title('Historical Daily Closing Prices of AT&T and Verizon')
        plt.legend()
        plt.grid()
        plt.show()

    def get_symbol(self, s):
      params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': s,
            'apikey': self.api_key,
            'outputsize': 'full',
            'datatype': 'csv'
              }
      response = requests.get(self.base_url, params=params)
      csv_data = response.text

      data = pd.read_csv(StringIO(csv_data))

      data['timestamp'] = pd.to_datetime(data['timestamp'])  # Convert timestamp to datetime
      filtered_data = data[(data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)]

      return filtered_data


if __name__ == "__main__":
    symbols = ['T', 'VZ']
    start_date = '2019-01-01'
    end_date = '2022-06-01'

    extractor = ExtractData(symbols, start_date, end_date)
    extractor.query()
    df = extractor.stocks_df.set_index('timestamp')

