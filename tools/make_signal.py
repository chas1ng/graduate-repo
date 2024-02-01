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


def make_rand_chirp(dt, f0, f1, t1):
    t = np.linspace(0, t1, int(t1 / dt))
    chirp_signal = chirp(t, f0=f0, f1=f1, t1=t1, method='quadratic', phi=90)
    rand = np.sin((np.pi * t) / (t[-1] / 2))
    rand_chirp = chirp_signal * rand
    return rand_chirp


def make_chirp():
    pass


def make_rand():
    pass


if __name__ == '__main__':

    dt = 1E-4
    f0 = 15
    f1 = 25
    t1 = 0.1

    pass
