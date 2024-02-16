#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：YJC
@Date    ：2024/2/16 15:52 
@Purpose ：
"""

import matplotlib.pyplot as plt
import numpy as np

datas = np.load('..//datas//dataset//yjc-1-17-chirp.npy')

print(datas.shape)

plt.plot(datas[1, :, 90])
plt.show()

