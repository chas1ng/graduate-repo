#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：YJC
@Date    ：2024/2/8 20:06 
@Purpose ：
"""

from main import *


def main():
    train_iter, dev_iter, test_iter = get_dataset()
    model = load_model()
    start(model, train_data, dev_data, test_data, Zscore)


if __name__ == '__main__':
    main()
