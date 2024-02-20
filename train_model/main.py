import copy
import os
import numpy as np
import torch.optim
import importlib
from tools.tool import *


# @print_run_time
# def get_train_dataset():
#     DataLoader.drop_last = True
#     train = get_one(config['train_path'])
#     dev = get_one(config['dev_path'])
#     test = get_CFD_data(config['test_path'])
#
#     train = DataLoader(TransDataset(train), batch_size=config['batch_size'], shuffle=config['shuffle'])
#     dev = DataLoader(TransDataset(dev), batch_size=config['batch_size'], shuffle=config['shuffle'])
#     test = DataLoader(TransDataset(test), batch_size=config['batch_size'], shuffle=config['shuffle'])
#
#     return train, dev, test
#
#
# @print_run_time
# def get_test_dataset():
#     train = get_one(config['train_path'])
#     dev = get_one(config['dev_path'])
#     test = get_CFD_data(config['test_path'])
#
#     train = DataLoader(TransDataset(train), batch_size=len(train))
#     dev = DataLoader(TransDataset(dev), batch_size=len(dev))
#     test = DataLoader(TransDataset(test), batch_size=len(test))
#
#     return train, dev, test
#
#
# def get_CFD_data(path):
#     outs = []
#     force_len, move_len = config['force_len'], config['move_len']
#     # dongya = int(path.split('_')[2].strip('q'))
#     lens = force_len + move_len
#     move_data, force_data = np.load(f'datas//CFD_data//{path}')
#     # force_data = force_data * 5100 / dongya
#     for i in range(len(move_data) - move_len - force_len):
#         outs.append(np.concatenate([move_data[i + force_len:i + lens], force_data[i + move_len: i + lens + 1]]))
#     outs = np.array(outs).astype(np.float32)
#     print(f'get {path} data shape is ', outs.shape)
#     return outs
#
#
# def get_one(path):
#     outs = []
#     force_len, move_len = config['force_len'], config['move_len']
#     lens = force_len + move_len
#     for one_path in path:
#         out = []
#         for mo in range(1, 3):
#             move_data, force_data = np.load(f'datas//dataset//{one_path}_{mo}.npy')
#             for i in range(len(move_data) - move_len - force_len):
#                 out.append(np.concatenate([move_data[i + force_len:i + lens], force_data[i + move_len: i + lens + 1]]))
#         outs = outs + out
#     outs = np.array(outs).astype(np.float32)
#     print(f'get {path} data shape is ', outs.shape)
#     return outs
#     # return DataLoader(TransDataset(outs), batch_size=config['batch_size'], shuffle=config['shuffle'], drop_last=True)
#
#
# def get_Zscore():
#     moves, forces = np.load(f"datas//dataset//{config['train_path'][0]}.npy")
#     return moves.mean(), moves.std(), forces.mean(), forces.std()
#


@print_run_time
def main():
    train_iter, dev_iter, test_iter = get_dataset()
    train_iter, dev_iter, test_iter = train_iter, dev_iter, test_iter
    model = load_model().to(device)
    start(model, train_iter, dev_iter, test_iter)


if __name__ == '__main__':
    # Zscore = np.load('datas//static//Z_source.npy')
    # config
    config['train'] = True
    config['reTrain'] = True
    config['iter'] = 'dev'
    config['lr'] = 0.0001
    config['use_model'] = 'LSTM'
    main()
