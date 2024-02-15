# -*- coding: utf-8 -*-
"""
Created on 2024/1/25 

@author: YJC

Purpose：
"""

# 方法确定后写到tool里面去

# Pod矩阵来源于位移，只对位移做分析

import numpy as np
import os
import warnings
import torch

warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

comp = np.load('E:\ROM\datas\static\move_comp.npy')


def train_pod():
    U, s, V = np.linalg.svd(base_data)
    k = s.sum()
    print(k)
    m = 0
    for i in range(len(s)):
        m += s[i]
        print(m / k)
        if m / k >= 0.99:
            print(i + 1)
            break
    print(np.expand_dims(U[0], axis=1).shape)
    return U, s, V


def SVD_try():
    # 计算
    U, s, V = train_pod()

    # 还原并对比
    # ----------重构-------------
    for test in [test1_data, test2_data, test3_data, test4_data]:
        S = 4
        R = []
        for i in range(100):
            snapshot_i = test[:, i]
            reconstruction = np.zeros_like(snapshot_i)
            # print(snapshot_i.shape)
            for s in range(S):
                # print(np.dot(U[:, s], snapshot_i))
                reconstruction = reconstruction + np.dot(U[:, s], snapshot_i) * U[:, s]
            R.append(reconstruction)

    R = np.array(R)
    print(R.shape)
    test = np.swapaxes(test, 0, 1)
    print(test.shape)


def PCA_try():
    # todo 先把除了误差以外的流程搞完
    estimator = PCA(n_components=4)
    pca_Y_train = estimator.fit_transform(base_data)
    print(type(estimator))
    print(base_data.shape)
    print(pca_Y_train.shape)

    pca_Y_test1 = estimator.transform(test1_data)
    print(pca_Y_test1.shape)

    comp = estimator.components_
    np.save('../datas/static/move_comp.npy', comp)
    print(comp.shape)
    print(type(comp))
    pca_X_train = np.matmul(pca_Y_train, comp)

    print(pca_X_train.shape)
    new_X = np.dot(base_data, np.linalg.pinv(comp))

    import matplotlib.pyplot as plt

    for i in range(len(pca_Y_train[0])):
        plt.figure(figsize=(18, 6))
        plt.plot(new_X[:, i], color='r', label='base')
        plt.plot(pca_Y_train[:, i], color='b', label='back')
        plt.legend()
        plt.show()
    print('________________________')
    for i in range(len(base_data[0])):
        plt.figure(figsize=(18, 6))
        plt.plot(base_data[:, i], color='r', label='base')
        plt.plot(pca_X_train[:, i], color='b', label='back')
        plt.legend()
        plt.show()


def Use_PCA(data):

    new_data = pca(data)

    print(data.shape, new_data.shape)

    test_data = un_pca(new_data)

    plt.plot(new_data[:, 0])
    plt.plot(new_data[:, 1])
    plt.plot(new_data[:, 2])
    plt.plot(new_data[:, 3])
    plt.show()

    for i in range(121):
        plt.figure(figsize=(18, 6))
        plt.plot(data[:, i], color='r', label='base')
        plt.plot(test_data[:, i], color='b', label='back')
        plt.legend()
        plt.show()


def pca(data):
    new_data = []
    if len(data.shape) == 3:
        data = data.detach().numpy()
        for i in range(data.shape[0]):
            new_data.append(np.dot(data[i], np.linalg.pinv(comp)) / 0.22347653)
        return torch.tensor(new_data)
    return np.dot(data, np.linalg.pinv(comp)) / 0.22347653


def un_pca(data):
    return np.matmul(data * 0.22347653, comp)


if __name__ == '__main__':
    base_path = '..//datas//dataset//5100_0.1_0.02.npy'
    test1_path = '..//datas//dataset//yjc-1-17-chirp.npy'
    test2_path = '..//datas//dataset//yjc-1-17-chirp-%10.npy'
    test3_path = '..//datas//dataset//yjc-1-17-rand-chirp.npy'
    test4_path = '..//datas//dataset//yjc-1-17-rand-chirp-%10.npy'

    # base_data = np.load(base_path)[0]
    # test1_data = np.load(test1_path)[0]
    # test2_data = np.load(test2_path)[0]
    # test3_data = np.load(test3_path)[0]
    # test4_data = np.load(test4_path)[0]

    # print(base_data.shape, test1_data.shape, test2_data.shape, test3_data.shape, test4_data.shape)
    #
    # CFD_data_1 = np.load('..//datas//CFD_data//m0.499_a0.1_q5171_nastran_metric.npy')[0]
    # CFD_data_2 = np.load('..//datas//CFD_data//m0.499_a0.1_q5600_nastran_metric.npy')[0]
    # CFD_data_3 = np.load('..//datas//CFD_data//m0.499_a0.1_q5610_nastran_metric.npy')[0]
    CFD_data_4 = np.load('../datas/CFD_data/n0.499_a0.1_q5700_nastran_metric.npy')[0]
    # CFD_data_5 = np.load('..//datas//dataset//yjc-1_2_donya_5100_0.1_1e-4.npy')[0]
    # SVD_try()
    # PCA_try()
    net_data = np.load('moves.npy')

    Use_PCA(net_data)
    # print(np.max(pca(base_data)))     归一化超参数
