# -*- coding: utf-8 -*-
__author__ = 'huangyf'

import numpy as np
import pandas as pd
import tushare as ts
from sklearn.preprocessing import MinMaxScaler


class Market(object):
    def __init__(self, args):
        self.codes = args.codes
        self.start_date = args.start
        self.end_date = args.ends
        self.original_stocks_data = dict()
        self.scaled_stocks_data = dict()
        self.dates = []
        self.action_space = 0
        self.observation_space = 0

        self.train_test_split = 0.7

        self.current_date = None
        self.next_date = None
        self.iter_date = None

        self.current_cash = args.cash
        self.init_cash = args.cash
        self.reward = 0

        self._init_stocks_data()

    def reset(self, mode='train'):
        self.iter_date = iter(self.train_dates) if mode == 'train' else iter(self.test_dates)
        self.current_date, self.next_date = next(self.iter_date), next(self.iter_date)
        return self._get_scaled_stocks_data_from_date(self.current_date)

    def step(self):
        pass

    def _init_stocks_data(self):
        columns = ['open', 'high', 'low', 'close', 'volume']
        dates_set = set()
        for code in self.codes:
            stock = ts.get_k_data(code, self.start_date, self.end_date)
            stock_date = stock['date'].as_matrix().tolist()
            stock_data = stock[columns].as_matrix()
            dates_set.union(stock_date)
            scale_stock_data = MinMaxScaler().fit_transform(stock_data)
            self.original_stocks_data[code] = pd.DataFrame(data=stock_data, index=stock_date, columns=columns)
            self.scaled_stocks_data[code] = pd.DataFrame(data=scale_stock_data, index=stock_date, columns=columns)

        self.dates = sorted(list(dates_set))

        for code in self.codes:
            self.original_stocks_data[code] = self.original_stocks_data[code].reindex(self.dates, method='bfill')
            self.scaled_stocks_data[code] = self.scaled_stocks_data[code].reindex(self.dates, method='bfill')

        self.train_dates = self.dates[:int(len(self.dates) * self.train_test_split)]
        self.test_dates = self.dates[int(len(self.dates) * self.train_test_split):]

    def _get_scaled_stocks_data_from_date(self, date):
        stocks_data = []
        for code in self.codes:
            data = self.scaled_stocks_data[code].loc[date].as_matrix().tolist()
            stocks_data.extend(data)
        return stocks_data
