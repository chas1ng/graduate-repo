# -*- coding: utf-8 -*-
"""
Created on 2024/2/1 

@author: YJC

Purpose：
"""
import matplotlib.pyplot as plt

from tools.tool import *


def _get_signal():
    rand_chirp_data = get_rand_chirp(t, f0, f1, t1)
    chirp_data = chirp(t, f0=f0, f1=f1, t1=t1, method='quadratic', phi=90)

    plt.plot(chirp_data)
    plt.plot(rand_chirp_data)
    plt.show()

    return rand_chirp_data, chirp_data


def _mix(data, len_time, start_time=50):
    """
    :param data:
    :param len_time: 间隔时间
    :return:
    """
    lens = len(data) * 2 + start_time * 2 - len_time
    mix_data = np.zeros([5, lens])

    for i in range(lens):
        mix_data[0, i] = (i + 1) * dt
    for i in range(len(data)):
        mix_data[1, i + start_time] = data[i]
    for i in range(len(data)):
        mix_data[2, i + start_time + len(data) - len_time] = data[i] * 0.5
    return mix_data


if __name__ == '__main__':
    dt = 1E-4
    f0 = 15
    f1 = 30
    t1 = 0.1
    mix = 800
    t = np.linspace(0, t1, int(t1 / dt))

    rand_c, c = _get_signal()

    new_data = _mix(rand_c, mix)

    plt.plot(new_data[1])
    plt.plot(new_data[2])
    plt.show()

    write_new_signal(new_data[1:] * 0.02, new_data[0], f"..//datas//signal//rand_chirp_{f0}_{f1}_{t1}_{mix}.dat")
