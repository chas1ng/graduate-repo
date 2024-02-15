# -*- coding: utf-8 -*-
"""
Created on 2023/12/19 

@author: YJC

Purpose：
"""

import numpy as np
import matplotlib.pyplot as plt
# from scipy.fft import fft, fftfreq

def FFT(signal, dt:float=1e-3, log:bool=True, Draw:bool=True):
    N = len(signal)
    yf = np.fft.fft(signal)
    xf = np.fft.fftfreq(N, dt)[:N // 2]
    print(xf.shape)
    plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))
    plt.xlabel('Frequency (B:Hz)')
    plt.ylabel('Amplitude (A)')
    if log:
        plt.yscale('log')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    import numpy as np

    data1 = np.load('..//datas//CFD_data//m0.499_a0.1_q5171_nastran_metric.npy')[0]
    data2 = np.load('..//datas//CFD_data//m0.499_a0.1_q5600_nastran_metric.npy')[0]
    data3 = np.load('..//datas//CFD_data//m0.499_a0.1_q5700_nastran_metric.npy')[0]

    base_data = np.load('..//datas//dataset//yjc-1-17-chirp-%10.npy')[0]

    print(data1.shape, data2.shape, data3.shape, base_data.shape)
    # 对位移的频域做计算
    dt = 1E-4
    FFT(data1[:, 10], dt)
    FFT(data1[:, 20], dt)
    FFT(data1[:, 30], dt)
    FFT(data1[:, 40], dt)
    FFT(data1[:, 50], dt)
    FFT(data1[:, 60], dt)
    FFT(data1[:, 70], dt)
    FFT(data1[:, 80], dt)
    FFT(data1[:, 90], dt)
    FFT(data1[:, 100], dt)
    FFT(data1[:, 110], dt)
    FFT(data1[:, 120], dt)

