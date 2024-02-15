# -*- coding: utf-8 -*-
"""
Created on 2024/1/15 

@author: YJC

Purpose：原始数据转numpy
"""

import numpy as np
from pathlib import Path
import os.path as path
import matplotlib.pyplot as plt
from configparser import ConfigParser


def trans(file_name: str) -> np.ndarray:
    nodeTi = 1
    Moves, Forces = [], []

    while True:
        print(f'\rload file: {nodeTi}', end='', flush=True)

        nodeMovePath = f'{file_name}//dyAE//nodemove_{nodeTi}.dat'
        nodeForcePath = f'{file_name}//dyAE//nodeforce_{nodeTi}.dat'

        if not path.isfile(nodeForcePath) or not path.isfile(nodeMovePath): break

        moves, forces = open(nodeMovePath, 'r').readlines(), open(nodeForcePath, 'r').readlines()
        Moves.append([[eval(move_line.split()[i]) for i in range(1, 4)] for move_line in moves])
        Forces.append([[eval(force_line.split()[i]) for i in range(1, 4)] for force_line in forces])
        # 可能出现下载错误的情况
        if len(Forces[-1]) != 121 or len(Moves[-1]) != 121: break
        nodeTi += 1

    datas = np.array([Moves, Forces]).astype(np.float32)
    print('\nGet datas: ', datas[0].shape, datas[1].shape)
    return datas


def draw(path):
    data = np.load(f'..//datas//dataset//{path}.npy')
    print(data.shape)
    plt.plot(data[0, :, 90], color='r')
    plt.show()

    plt.plot(data[1, :, 90], color='r')
    plt.show()


if __name__ == '__main__':
    root_file = 'E://yjc//Trip结果'
    file_name = 'yjc_2_1_rand_chirp'

    numpy_data = trans(f'{root_file}//{file_name}')[:, :, :, -2]
    np.save(f'..//datas//dataset//{file_name}', numpy_data)
    draw(file_name)
