# -*- coding: utf-8 -*-
__author__ = 'huangyf'

import tushare as ts


class Market(object):
    def __init__(self, args):
        self.codes = args.codes
        self.start_date = args.start
        self.end_date = args.ends
        self.stocks_data = []
        self.action_space = 0
        self.observation_space = 0

    def reset(self):
        pass

    def step(self):
        pass

    def get_observation(self, date):
        pass

    def _init_data(self):
        pass
