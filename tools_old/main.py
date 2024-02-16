import copy
import os
import numpy as np
import torch.optim
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tools.dataset import TransDataset
from tools.tool import print_run_time, get_config
from tools.Pod import pca
import importlib


@print_run_time
def get_train_dataset():
    DataLoader.drop_last = True
    train = get_one(config['train_path'])
    dev = get_one(config['dev_path'])
    test = get_CFD_data(config['test_path'])

    train = DataLoader(TransDataset(train), batch_size=config['batch_size'], shuffle=config['shuffle'])
    dev = DataLoader(TransDataset(dev), batch_size=config['batch_size'], shuffle=config['shuffle'])
    test = DataLoader(TransDataset(test), batch_size=config['batch_size'], shuffle=config['shuffle'])

    return train, dev, test


@print_run_time
def get_test_dataset():
    train = get_one(config['train_path'])
    dev = get_one(config['dev_path'])
    test = get_CFD_data(config['test_path'])

    train = DataLoader(TransDataset(train), batch_size=len(train))
    dev = DataLoader(TransDataset(dev), batch_size=len(dev))
    test = DataLoader(TransDataset(test), batch_size=len(test))

    return train, dev, test


def get_CFD_data(path):
    outs = []
    force_len, move_len = config['force_len'], config['move_len']
    # dongya = int(path.split('_')[2].strip('q'))
    lens = force_len + move_len
    move_data, force_data = np.load(f'datas//CFD_data//{path}')
    # force_data = force_data * 5100 / dongya
    for i in range(len(move_data) - move_len - force_len):
        outs.append(np.concatenate([move_data[i + force_len:i + lens], force_data[i + move_len: i + lens + 1]]))
    outs = np.array(outs).astype(np.float32)
    print(f'get {path} data shape is ', outs.shape)
    return outs


def get_one(path):
    outs = []
    force_len, move_len = config['force_len'], config['move_len']
    lens = force_len + move_len
    for one_path in path:
        out = []
        for mo in range(1, 3):
            move_data, force_data = np.load(f'datas//dataset//{one_path}_{mo}.npy')
            for i in range(len(move_data) - move_len - force_len):
                out.append(np.concatenate([move_data[i + force_len:i + lens], force_data[i + move_len: i + lens + 1]]))
        outs = outs + out
    outs = np.array(outs).astype(np.float32)
    print(f'get {path} data shape is ', outs.shape)
    return outs
    # return DataLoader(TransDataset(outs), batch_size=config['batch_size'], shuffle=config['shuffle'], drop_last=True)


def get_Zscore():
    moves, forces = np.load(f"datas//dataset//{config['train_path'][0]}.npy")
    return moves.mean(), moves.std(), forces.mean(), forces.std()


def load_model():
    use_model = config['use_model']
    model = importlib.import_module(f'models.model_{use_model}').Model(config)
    print(f"Loading {config['use_model']} model:")
    print(model)
    return model


def start_train(model, train_iter, dev_iter, test_iter, Zscore):
    # todo retrain
    L1 = 2000000
    L2 = 1000000

    train_save_path = f"models//save_model//{config['use_model']}_{config['move_len']}_{config['force_len']}_train"
    test_save_path = f"models//save_model//{config['use_model']}_{config['move_len']}_{config['force_len']}_test"

    model.load_state_dict(torch.load(train_save_path))
    optimizer = None
    criterion = None
    if config['optim'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], )
        # weight_decay=config['init_weight_decay'])
    if config['loss'] == 'MAE':
        criterion = torch.nn.MSELoss()

    for epoch in range(config['epochs']):
        print(f"\r## The {epoch} Epoch, ALL {config['epochs']} Epochs ! ##", end='')
        loss1 = loss2 = 0

        model.train()
        for i, batch in enumerate(train_iter):
            inputs, targets = batch[:, :-1], batch[:, -1]
            inputs = pca(inputs)
            targets = trans_force(targets)
            outs = model(inputs)

            optimizer.zero_grad()
            loss = criterion(outs, targets)
            loss.backward()
            optimizer.step()
            loss1 += loss.item()

        model.eval()
        for i, batch in enumerate(dev_iter):
            inputs, targets = batch[:, :-1], batch[:, -1]

            inputs = pca(inputs)
            targets = trans_force(targets)

            outs = model(inputs)
            loss = criterion(outs, targets)
            loss2 += loss.item()

        if epoch % 10 == 0:
            print(f'  train_loss: {loss1 / len(train_iter)}, test_loss: {loss2 / len(dev_iter)}')

        if L1 > loss1:
            L1 = loss1
            torch.save(model.state_dict(), train_save_path)
        if L2 > loss2:
            L2 = loss2
            torch.save(model.state_dict(), test_save_path)
    pass


def start_test(model, train_iter, dev_iter, test_iter, Zscore):
    criterion = torch.nn.MSELoss()
    train_save_path = f"models//save_model//{config['use_model']}_{config['move_len']}_{config['force_len']}_train"
    test_save_path = f"models//save_model//{config['use_model']}_{config['move_len']}_{config['force_len']}_test"

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
    for i, batch in enumerate(test_iter):
        inputs, targets = batch[:, :-1], batch[:, -1]
        inputs = pca(inputs)
        targets = trans_force(targets)
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
    # 临时
    for i in range(80, 121):
        plt.figure(figsize=(12, 6))
        plt.title(f'Point: {i}')
        # plt.plot(inputs[:, -1, 0].detach(), label='m1')
        # plt.plot(inputs[:, -1, 1].detach(), label='m2')
        # plt.plot(inputs[:, -1, 2].detach(), label='m3')
        # plt.plot(inputs[:, -1, 3].detach(), label='m4')
        # plt.plot(old_outs[:, i].detach(), label='outs')
        # plt.plot(old_targets[:, i].detach() * 5700 / 5100, label='targets')
        plt.plot(t_outs[:, i].detach(), label='t_outs')
        plt.plot(t_targets[:, i].detach() * 5700 / 5100, label='t_targets')
        # plt.plot(inputs[:, -1, 90], label='Move')
        plt.legend()
        plt.show()


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


@print_run_time
def main():
    train_iter, dev_iter, test_iter = get_train_dataset()
    train_data, dev_data, test_data = get_test_dataset()

    # print(f'Zscore: {Zscore}')
    # define_dict()
    # load_preEmbedding()
    # update_arguments()

    # save_arguments()
    model = load_model()

    # 保存模型时，保存相应的参数.txt
    # start_train(model, train_iter, dev_iter, test_iter, Zscore)
    start_test(model, train_data, dev_data, test_data, Zscore)


if __name__ == '__main__':
    Zscore = np.load('datas//static//Z_source.npy')
    config = get_config('datas//Config//config.yml')
    main()
