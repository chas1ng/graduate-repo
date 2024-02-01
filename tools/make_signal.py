#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：YJC
@Date    ：2024/2/1 8:16 
@Purpose ：
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
from tools.tool import *


def _get_signal_mix():
    # date: 2_1
    rand_chirp_data = get_rand_chirp(t, f0, f1, t1)
    chirp_data = chirp(t, f0=f0, f1=f1, t1=t1, method='quadratic', phi=90)
    #
    plt.figure(figsize=(16, 8))
    plt.subplot(2, 1, 1)
    plt.title(f'Make Signal')
    plt.plot(rand_chirp_data, label='rand_chirp')
    plt.plot(chirp_data, label='chirp')
    plt.subplot(2, 1, 2)
    a, b = FFT(chirp_data, dt)
    print(a.shape, b.shape)
    a1, b1 = FFT(rand_chirp_data, dt)

    plt.plot(a1, b1, label='rand_chirp')
    plt.plot(a, b, label='chirp')

    plt.yscale('log')
    plt.legend()
    plt.show()
    return rand_chirp_data, chirp_data


def _mix(data, len_time):
    start_time = 50
    lens = len(data) * 2 + start_time * 2 - len_time
    mix_data = np.zeros([5, lens])
    print(data.shape)
    for i in range(lens):
        mix_data[0, i] = int(i * dt)
    for i in range(len(data)):
        mix_data[1, i + start_time] = data[i]
    for i in range(len(data)):
        mix_data[2, i + start_time + len(data) - len_time] = data[i] * 0.5
    return mix_data


if __name__ == '__main__':
    # 参数
    dt = 1E-4
    f0 = 15
    f1 = 30
    t1 = 0.1
    mix = 800
    t = np.linspace(0, t1, int(t1 / dt))

    r_c, c = _get_signal_mix()
    print(r_c.shape, c.shape)
    new_data = _mix(c, mix)

    plt.plot(new_data[1])
    plt.plot(new_data[2])
    plt.show()

    write_new_signal(new_data[1:], new_data[0], f'..//datas//chirp_{f0}_{f1}_{t1}_{800}.dat')
