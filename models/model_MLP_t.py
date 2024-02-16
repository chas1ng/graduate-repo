# -*- coding: utf-8 -*-
"""
Created on 2024/1/17 

@author: YJC

Purpose：
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import random
from tools.tool import get_config, print_run_time


class Model(nn.Module):

    @print_run_time
    def __init__(self, config):
        super(Model, self).__init__()

        seed_num = config['seed_num']
        torch.manual_seed(seed_num)
        random.seed(seed_num)

        self.config = config
        self.dropout = config['dropout']
        self.input_size = config['input_size'] * (config['move_len'] + config['force_len'])
        self.output_size = config['output_size']
        self.hidden_size = config['MLP_hidden_size']
        self.layers_dim = config['MLP']['layers_dim']

        self.linear = nn.Linear(self.input_size, self.output_size)
        # self.nets = nn.Sequential(
        #     nn.BatchNorm1d(self.input_size),
        #     nn.Linear(self.input_size, 300),
        #     nn.Sigmoid(),
        #     nn.BatchNorm1d(300),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(300, 300),
        #     nn.Sigmoid(),
        #     nn.BatchNorm1d(300),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(300, 300),
        #     nn.Sigmoid(),
        #     nn.BatchNorm1d(300),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(300, 300),
        #     nn.Sigmoid(),
        #     nn.BatchNorm1d(300),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(300, 300),
        #     nn.Sigmoid(),
        #     nn.Linear(300, self.output_size)
        # )
        self.net_base = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.Sigmoid(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Sigmoid(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Sigmoid(),
            nn.Linear(self.hidden_size, self.output_size)
        )
        if config['init_weight']:
            print('Init_ing W ')

    def forward(self, inputs):
        b, M, Dim = inputs.shape
        inputs = inputs.reshape(b, M * Dim)
        outs = self.net_base(inputs)
        return outs

if __name__ == '__main__':
    config = get_config('..//datas//Config//config.yml')
    ne = Model(config)
    print(ne)
    test_data = torch.zeros([10, 6, 121])

    test_outs = ne(test_data)
    print(test_outs.shape)
