# -*- coding: utf-8 -*-
"""
Created on 2024/1/20 

@author: YJC

Purpose：
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tools.tool import print_run_time, get_config


@print_run_time
def trans(file_name: str) -> np.ndarray:
    nodeTi = 1
    Moves, Forces = [], []
    print('Load from dyAE file')
    while True:
        print(f'\rload file: {nodeTi}', end='', flush=True)
        try:
            move = open(file_name + '//dyAE//nodemove_{t}'.format(t=nodeTi), 'r').readlines()
            force = open(file_name + '//dyAE//dy_load_{t}.dat'.format(t=nodeTi), 'r').readlines()
            Moves.append(read_data(move))
            Forces.append(read_data(force, Move=False))
            nodeTi += 1
        except:
            print('\nLoad all steps: ', nodeTi)
            break
    data = np.array([Moves, Forces])
    return data

def read_data(readlines, axis=-1, Move=True):
    datas = []
    for line in readlines:
        if len(line) < 2: break
        if Move:
            datas.append(float(line.split()[axis]))
        else:
            datas.append(float(line.split(',')[axis]))
    return datas


if __name__ == '__main__':
    root_file = 'E://yjc//Trip结果'
    # root_file = 'E://yjc//trip_nastran'
    file_name = 'st_old_m_m_a0.1-un-mb-test5-q5700-t0.0001'

    numpy_data = trans(f'{root_file}//{file_name}')
    print(numpy_data.shape)
    np.save(f'..//datas//CFD_data//{file_name}', numpy_data)
    plt.plot(numpy_data[0, :, -1], color='r')
    plt.show()

    plt.plot(numpy_data[1, :, -1], color='r')
    plt.show()
