#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：YJC
@Date    ：2024/2/1 8:33 
@Purpose ： 大部分用到的函数，按照模型训练的顺序放置。
"""

import yaml
import time
import numpy as np
import torch
from scipy.signal import chirp
from torch.utils.data import Dataset, DataLoader

# import matplotlib.pyplot as plt


"""
数据部分：
（1）读取原始数据
（2）读取numpy数据
（3）数据转Dataloader类
（4）数据转换
（5）数据分析/结果分析
（6）生成激励信号
（7）
"""


def dataloader_numpy_data(path, is_part=False, is_ouhe=False):
    # 包含最简单的读取，拆分的数据，耦合结果的数据, 返回dataloader类
    if is_part:
        return 1
    if is_ouhe:
        return 0
    else:
        return np.load(f'D://Code//graduate-repo//datas//dataset//{path}.npy')


"""
模型部分：
（1）
"""

"""
耦合部分：
（1）
"""


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


def get_config(path):
    return yaml.load(open(path, 'r'), Loader=yaml.FullLoader)


# def
def get_rand_chirp(t, f0, f1, t1):
    chirp_signal = chirp(t, f0=f0, f1=f1, t1=t1, method='quadratic', phi=90)
    rand = np.sin((np.pi * t) / (t[-1] / 2))
    rand_chirp = chirp_signal * rand
    return rand_chirp


def write_new_signal(signal, t: list, name):
    # 格式没有细写
    trans = lambda num: f'{num:.6E}'
    print(len(signal[0]), len(t))
    with open(name, 'w') as F:
        for i in range(len(t)):
            F.write(
                f'{trans(t[i])}   {trans(signal[0][i])}   {trans(signal[1][i])}   {trans(signal[2][i])}   {trans(signal[3][i])}\n')
        F.close()


comp = np.zeros([5])


def pca(data):
    # https://blog.csdn.net/u012162613/article/details/42192293
    new_data = []
    if len(data.shape) == 3:
        data = data.detach().numpy()
        for i in range(data.shape[0]):
            new_data.append(np.dot(data[i], np.linalg.pinv(comp)))
        return torch.tensor(new_data)
    return np.dot(data, np.linalg.pinv(comp))


def un_pca(data):
    return np.matmul(data, comp)


"""
杂类
"""


def print_run_time(func):
    def wrapper(*args, **kw):
        local_time = time.time()
        print(f'\n--------start: {func.__name__}--------')
        res = func(*args, **kw)
        print(f'-------- end:  {func.__name__} used time: {time.time() - local_time}\n')
        return res

    return wrapper
