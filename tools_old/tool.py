# -*- coding: utf-8 -*-
"""
Created on 2024/1/16 

@author: YJC

Purpose：
"""
import time
import yaml
import numpy as np
from scipy.signal import chirp
import matplotlib.pyplot as plt


def print_run_time(func):
    def wrapper(*args, **kw):
        local_time = time.time()
        print(f'\n--------start: {func.__name__}--------')
        res = func(*args, **kw)
        print(f'-------- end:  {func.__name__} used time: {time.time() - local_time}\n')
        return res
    return wrapper

def get_config(path):
    return yaml.load(open(path, 'r'), Loader=yaml.FullLoader)


def get_rand_chirp(t, f0, f1, t1):
    chirp_signal = chirp(t, f0=f0, f1=f1, t1=t1, method='quadratic', phi=90)
    rand = np.sin((np.pi * t) / (t[-1] / 2))
    rand_chirp = chirp_signal * rand
    return rand_chirp

def write_new_signal(signal, t, name):

    trans = lambda num: f'{num:6E}'

    with open(name, 'w') as F:
        for i in range(len(t)):
            F.write(f"{trans(t[i])}   {trans(signal[0][i])}   {trans(signal[1][i])}   {trans(signal[2][i])}   {trans(signal[3][i])}\n")
        F.close()