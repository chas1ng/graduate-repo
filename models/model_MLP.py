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
        self.input_size = config['input_size']
        self.output_size = config['output_size']

        self.nets = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.input_size, 300),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(300, self.output_size)
        )
        if config['init_weight']:
            print('Init_ing W ')
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, inputs):
        b, M, Dim = inputs.shape
        inputs = inputs.view(b, M * Dim)
        inputs = self.dropout(inputs)
        outs = self.linear(inputs)
        return outs

if __name__ == '__main__':
    config = get_config('..//datas//Config//config.yml')
    ne = Model(config)
    print(ne)
    test_data = torch.zeros([10, 6, 121])

    test_outs = ne(test_data)
    print(test_outs.shape)
