#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：YJC
@Date    ：2024/1/31 20:29 
@Purpose ：
"""

print('hello world')
import numpy as np

k = np.load('m0.499_a0.1_q5100_mix.npy')
print(k.shape)


da1 = np.load('datas//dataset//yjc-1-17-chirp.npy')
da2 = np.load('datas//dataset//yjc-1-17-chirp-%10.npy')
da3 = np.load('datas//dataset//yjc-1-17-rand-chirp.npy')
da4 = np.load('datas//dataset//yjc-1-17-rand-chirp-%10.npy')

print(da1.shape, da2.shape, da3.shape, da4.shape)
