# -*- coding: utf-8 -*-
__author__ = 'huangyf'

import argparse

stock_codes = ["600036", "601328", "601998", "601398"]

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--codes', default=stock_codes, nargs='+')
parser.add_argument('-s', '--start', default='2008-01-01')
parser.add_argument('-e', '--end', default='2018-01-01')
