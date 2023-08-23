import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import requests
from io import StringIO
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np
from dotenv import load_dotenv
load_dotenv()


class DatasetClass:

    def __init__(self, symbol, start_date, end_date, sequence_length=60, validation=False,
                 test_size=0.1):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length
        self.validation = validation
        self.test_size = test_size
        self.base_url = 'https://www.alphavantage.co/query'
        self.api_key = os.environ['KEY']

        stocks_df = self.query()
        self.stocks_df = stocks_df[::-1].set_index('timestamp')
        self.normd_close = self.normalize()

        if validation:
            self.train, self.valid, self.test = self.split_data(self.normd_close)

            self.X_train, self.y_train, self.timestamps_x_train, \
                self.timestamps_y_train = self.get_sequence(self.train)

            self.X_valid, self.y_valid, self.timestamps_x_valid, \
                self.timestamps_y_valid = self.get_sequence(self.valid)

            self.X_test, self.y_test, self.timestamps_x_test, \
                self.timestamps_y_test = self.get_sequence(self.test)

        else:
            self.train, self.test = self.split_data(self.normd_close)

            self.X_train, self.y_train, self.timestamps_x_train, \
                self.timestamps_y_train = self.get_sequence(self.train)

            self.X_test, self.y_test, self.timestamps_x_test, \
                self.timestamps_y_test = self.get_sequence(self.test)

    def split_data(self, data):
        train_size = int((1 - self.test_size) * len(data))
        if self.validation:
            valid_size = int(0.5 * self.test_size * len(data))
            train, valid, test = data.iloc[:train_size], data.iloc[train_size:train_size + valid_size], data.iloc[
                                                                                                        train_size + valid_size:]
            return train, valid, test

        train, test = data.iloc[:train_size], data.iloc[train_size:]
        return train, test

    def get_sequence(self, data):
        X, Y = [], []
        time_x, time_y = [], []

        for i in range(len(data) - self.sequence_length):
            X.append(data.iloc[i:i + self.sequence_length])
            Y.append(data.iloc[i + self.sequence_length])

            time_x.append(data.index[i:i + self.sequence_length])
            time_y.append(data.index[i + self.sequence_length])

        X = np.array(X)
        Y = np.array(Y)
        time_x = np.array(time_x)
        time_y = np.array(time_y)

        return X, Y, time_x, time_y

    def normalize(self):
        prices = self.stocks_df['close']
        max_price = max(prices)
        min_price = min(prices)
        prices = (prices - min_price) / (max_price - min_price)
        return prices

    def query(self):
        stocks_df = pd.DataFrame({})

        stocks_df = self.get_symbol(self.symbol)

        return stocks_df

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
        filtered_data = data[(data['timestamp'] >= self.start_date) & (data['timestamp'] <= self.end_date)]

        return filtered_data


if __name__ == "__main__":
    symbol = 'AAPL'
    start_date = '2019-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    sequence_length = 10
    data = DatasetClass(symbol, start_date, end_date, sequence_length=sequence_length)
