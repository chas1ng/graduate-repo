#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：YJC
@Date    ：2024/2/1 8:33 
@Purpose ：
"""

import numpy as np
from scipy.signal import chirp
import matplotlib.pyplot as plt


def FFT(signal, dt: float = 1e-3, log: bool = True):
    N = len(signal)
    yf = np.fft.fft(signal)
    xf = np.fft.fftfreq(N, dt)[: N // 2]
    return xf, 2.0 / N * np.abs(yf[0: N // 2])
    # 画图示例
    # plt.plot()
    # plt.xlabel('Frequency (B:Hz)')
    # plt.ylabel('Amplitude (A)')
    #
    # if log:
    #     plt.yscale('log')


# def
def get_rand_chirp(t, f0, f1, t1):
    chirp_signal = chirp(t, f0=f0, f1=f1, t1=t1, method='quadratic', phi=90)
    rand = np.sin((np.pi * t) / (t[-1] / 2))
    rand_chirp = chirp_signal * rand
    return rand_chirp


def write_new_signal(signal, t: list, name):
    # 格式没有细写
    print(len(signal[0]), len(t))
    with open(name, 'w') as F:
        for i in range(len(t)):
            F.write(f'{t[i]} {signal[0][i]} {signal[1][i]} {signal[2][i]} {signal[3][i]}\n')
        F.close()
