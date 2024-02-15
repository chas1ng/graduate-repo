# -*- coding: utf-8 -*-
"""
Created on 2024/1/22 

@author: YJC

Purpose：
"""

import numpy
import numpy as np
import matplotlib.pyplot as plt

file_path = 'yjc-1-17-rand-chirp'

base = np.load(f'..//datas//dataset//{file_path}.npy')
d1 = np.load(f'..//datas//dataset//{file_path}_1.npy')
d2 = np.load(f'..//datas//dataset//{file_path}_2.npy')
d3 = np.load(f'..//datas//dataset//{file_path}_3.npy')
d4 = np.load(f'..//datas//dataset//{file_path}_4.npy')
point = 90
print(base.shape)
plt.figure(figsize=(16, 8))
plt.plot(base[1][:, point])
plt.plot(d1[1][:, point])
plt.plot(d2[1][:, point])
plt.plot(d3[1][:, point])
plt.plot(d4[1][:, point])

plt.show()
