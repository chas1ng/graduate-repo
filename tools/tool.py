#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：YJC
@Date    ：2024/2/1 8:33 
@Purpose ：
"""

import numpy as np
import matplotlib.pyplot as plt


def FFT(signal, dt: float = 1e-3, log: bool = True):
    N = len(signal)
    yf = np.fft.fft(signal)
    xf = np.fft.fftfreq(N, dt)[: N // 2]
    return xf, 2.0 / N * np.abs(yf[0: N // 2])
    # plt.plot()
    # plt.xlabel('Frequency (B:Hz)')
    # plt.ylabel('Amplitude (A)')
    #
    # if log:
    #     plt.yscale('log')
