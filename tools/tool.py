#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：YJC
@Date    ：2024/2/1 8:33 
@Purpose ： 大部分用到的函数，按照模型训练的顺序放置。
"""

import os
import shutil

import yaml
import time
import numpy as np
import torch
import importlib
import matplotlib.pyplot as plt
import copy

from scipy.signal import chirp
from torch.utils.data import Dataset, DataLoader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def print_run_time(func):
    # 记录运行时间
    def wrapper(*args, **kw):
        local_time = time.time()
        print(f'\n--------start: {func.__name__}--------')
        res = func(*args, **kw)
        print(f'--------end:   {func.__name__} used time: {time.time() - local_time}\n')
        return res

    return wrapper


def get_config(path):
    return yaml.load(open(path, 'r'), Loader=yaml.FullLoader)


config = get_config('..//datas//config.yml')
if not config['Pod']:
    config['input_size'] = 121
Zscore = np.load(f"{config['project_path']}datas//static//Z_source.npy")
comp = np.load(f"{config['project_path']}datas//static//move_comp.npy")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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


class TransDataset(Dataset):

    def __init__(self, data):
        self.dataset = data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]


def trans_numpy_data(path, is_part=True, is_ouhe=False):
    # 包含最简单的读取，拆分的数据，耦合结果的数据, 返回dataloader类
    move_data, force_data = np.load(f"{config['project_path']}datas//dataset//{path}.npy")
    force_len, move_len = config['force_len'], config['move_len']
    lens = force_len + move_len

    outs = []

    if is_part:
        # 按照模态取点，把末尾的部分去掉
        use_len = move_data.shape[0] // 4
        for i in range(4):
            move = move_data[use_len * i:use_len * (i + 1) - 100]
            force = force_data[use_len * i:use_len * (i + 1) - 100]
            weight = np.zeros([1, 121])
            for j in range(len(move) - move_len - force_len):
                outs.append(np.concatenate(
                    [move[j + force_len:j + lens], force[j + move_len: j + lens + 1], weight + 1 / (i + 1)]))

        [outs.insert(0, outs[0]) for i in range(20)]
        outs = np.array(outs).astype(np.float32)
        print(f'get {path} data shape is ', outs.shape)
        # 用来检查切块的效果
        # plt.plot(outs[:, -1, 90])
        # plt.show()
        return outs
    elif is_ouhe:
        weight = np.zeros([1, 121])
        for i in range(len(move_data) - move_len - force_len):
            outs.append(
                np.concatenate([move_data[i + force_len:i + lens], force_data[i + move_len: i + lens + 1], weight]))
        outs = np.array(outs).astype(np.float32)
        print(f'get {path} data shape is ', outs.shape)
        return outs
    for i in range(len(move_data) - move_len - force_len):
        outs.append(np.concatenate([move_data[i + force_len:i + lens], force_data[i + move_len: i + lens + 1]]))
    outs = np.array(outs).astype(np.float32)
    print(f'get {path} data shape is ', outs.shape)
    return outs


@print_run_time
def get_dataset():
    # train
    train_data = []
    [train_data.append(trans_numpy_data(f'{path}')) for path in config['train_path']]
    train_data = np.concatenate(train_data, axis=0)
    print('get all train data: ', train_data.shape)
    # dev
    dev_data = []
    [dev_data.append(trans_numpy_data(f'{path}')) for path in config['dev_path']]
    dev_data = np.concatenate(dev_data, axis=0)
    print('get all dev data: ', dev_data.shape)
    # test
    test_data = [trans_numpy_data(f"{config['test_path']}", config, is_ouhe=True)]

    test_data = np.concatenate(test_data, axis=0)
    print('get all test data: ', test_data.shape)

    if config['train']:
        DataLoader.drop_last = True
        train = DataLoader(TransDataset(train_data), batch_size=config['batch_size'], shuffle=config['shuffle'])
        dev = DataLoader(TransDataset(dev_data), batch_size=config['batch_size'], shuffle=config['shuffle'])
        test = DataLoader(TransDataset(test_data), batch_size=config['batch_size'], shuffle=config['shuffle'])
        return train, dev, test
    else:
        train = DataLoader(TransDataset(train_data), batch_size=len(train_data))
        dev = DataLoader(TransDataset(dev_data), batch_size=len(dev_data))
        test = DataLoader(TransDataset(test_data), batch_size=len(test_data))
        return train, dev, test


def get_trans(data):
    # 输入应该是力
    if data.shape[1] != 121:
        print('wrong')
        return -1
    else:
        zscore = []
        for i in range(121):
            zscore.append([np.mean([data[:, i]]), np.std([data[:, i]])])
        # for i in range(121):
        #     data[:, i] = (data[:, i] - zscore[i][0]) / zscore[i][1]
        #     plt.plot(data[:, i])
        #     plt.show()
        np.save(f"{config['project_path']}datas//static//Z_source.npy", np.array(zscore))


def trans_force(data):
    for i in range(121):
        data[:, i] = (data[:, i] - Zscore[i][0]) / Zscore[i][1]
    return data


def un_force(data):
    print(data.shape)
    for i in range(121):
        data[:, i] = data[:, i] * Zscore[i][1] + Zscore[i][0]
        print(Zscore[i][1], Zscore[i][0])
    return data


"""
模型部分：
（1）
"""


def load_model():
    use_model = config['use_model']
    model = importlib.import_module(f'models.model_{use_model}').Model(config)
    print(f"Loading {config['use_model']} model:")
    print(model)
    return model


def start(model, train_iter, dev_iter, test_iter):
    use_model = config['use_model']
    move_len = config['move_len']
    force_len = config['force_len']

    train_save_path = f"{config['project_path']}models//save_model//{use_model}_{move_len}_{force_len}_train"
    test_save_path = f"{config['project_path']}models//save_model//{use_model}_{move_len}_{force_len}_test"

    if config['reTrain']:
        shutil.copyfile(train_save_path, train_save_path + '_base')
        model.load_state_dict(torch.load(train_save_path))
        L1 = 0.2
        L2 = 0.1
    else:
        L1 = 2
        L2 = 1
    optimizer = None
    criterion = None
    if config['optim'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], )
        # weight_decay=config['init_weight_decay'])
    if config['loss'] == 'MAE':
        criterion = torch.nn.MSELoss()

    # train
    if config['train']:

        for epoch in range(config['epochs']):
            print(f"\r## The {epoch} Epoch, ALL {config['epochs']} Epochs ! ##", end='')
            loss1 = 0
            loss2 = 0

            model.train()

            for i, batch in enumerate(train_iter):
                inputs, targets, weight = batch[:, :-2], batch[:, -2], batch[:, -1]
                if config['Pod']:
                    inputs = pca(inputs).to(device)
                targets = trans_force(targets).to(device)
                outs = model(inputs)

                optimizer.zero_grad()
                loss = criterion(outs, targets) * weight.to(device)
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                loss1 += loss.item()

            model.eval()
            for i, batch in enumerate(dev_iter):
                inputs, targets, weight = batch[:, :-2], batch[:, -2], batch[:, -1]

                inputs = pca(inputs).to(device)
                targets = trans_force(targets).to(device)

                outs = model(inputs)
                loss = criterion(outs, targets) * weight.to(device)
                loss = loss.mean()
                loss2 += loss.item()

            if epoch % 10 == 0:
                print(f'  train_loss: {loss1 / len(train_iter)}, test_loss: {loss2 / len(dev_iter)}')

            if L1 > loss1 / len(train_iter):
                L1 = loss1
                # torch.save(model.state_dict(), train_save_path)
            if L2 > loss2 / len(dev_iter):
                L2 = loss2
                # torch.save(model.state_dict(), test_save_path)
    else:
        criterion = torch.nn.MSELoss()
        if config['Test']['use_model']:
            model.load_state_dict(torch.load(train_save_path))
        else:
            model.load_state_dict(torch.load(test_save_path))
        print(test_save_path)
        model.eval()
        inputs = None
        outs = None
        targets = None
        loss = None
        if config['iter'] == 'train':
            use_iter = train_iter
        elif config['iter'] == 'dev':
            use_iter = dev_iter
        else:
            use_iter = test_iter

        for i, batch in enumerate(use_iter):
            inputs, targets, weight = batch[:, :-2], batch[:, -2], batch[:, -1]
            inputs = pca(inputs).to(device)
            targets = trans_force(targets).to(device)
            outs = model(inputs)
            loss = criterion(outs, targets)
        print(loss.item())
        print(inputs.shape)
        print(targets.shape)
        print(outs.shape)
        old_targets = copy.deepcopy(targets.detach())
        old_outs = copy.deepcopy(outs.detach())
        t_targets = un_force(targets)
        t_outs = un_force(outs)
        for i in range(80, 121):
            plt.figure(figsize=(12, 6))
            plt.title(f'Point: {i}')
            # plt.plot(inputs[:, -1, 0].detach(), label='m1')
            # plt.plot(inputs[:, -1, 1].detach(), label='m2')
            # plt.plot(inputs[:, -1, 2].detach(), label='m3')
            # plt.plot(inputs[:, -1, 3].detach(), label='m4')
            # plt.plot(old_outs[:, i].detach(), label='outs')
            # plt.plot(old_targets[:, i].detach() * 5700 / 5100, label='targets')
            plt.plot(t_outs[:, i].cpu().detach() * 5700 / 5100, label='t_outs')
            plt.plot(t_targets[:, i].cpu().detach(), label='t_targets')
            # plt.plot(inputs[:, -1, 90], label='Move')
            plt.legend()
            plt.show()


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


def pca(data):
    # https://blog.csdn.net/u012162613/article/details/42192293
    new_data = []
    if len(data.shape) == 3:
        data = data.detach().numpy()
        for i in range(data.shape[0]):
            new_data.append(np.dot(data[i], np.linalg.pinv(comp)).tolist())
        return torch.tensor(new_data)
    return np.dot(data, np.linalg.pinv(comp))


def un_pca(data):
    return np.matmul(data, comp)


"""
杂类
"""
