#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：YJC
@Date    ：2024/2/16 16:03 
@Purpose ：
"""

# 指定一个原始数据集

import numpy as np
import matplotlib.pyplot as plt

path = '..//datas//dataset//yjc-1-17-rand-chirp.npy'

datas = np.load(path)

print(datas.shape)
print(datas.shape[1])
plt.plot(datas[0, :, 80])
plt.show()

use_len = datas.shape[1] // 4
print(use_len)

plt.plot(datas[1, :use_len - 50, 90])
plt.show()

part1 = datas[0, use_len * 0:use_len * 1 - 50, 90]
part2 = datas[0, use_len * 1:use_len * 2 - 50, 90]
part3 = datas[0, use_len * 2:use_len * 3 - 50, 90]
part4 = datas[0, use_len * 3:use_len * 4 - 50, 90]

plt.plot(part1)
plt.plot(part2)
plt.plot(part3)
plt.plot(part4)
plt.show()

new_datas = []

for i in range(len(datas[0])):
    pass


def avoid_dataset():
    pass
