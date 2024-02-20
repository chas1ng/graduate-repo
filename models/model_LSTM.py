# -*- coding: utf-8 -*-
"""
Created on 2024/1/17 

@author: YJC

Purpose：
"""

import torch.nn as nn
import torch.nn.init as init
import random
from tools.tool import *


class Model(nn.Module):

    @print_run_time
    def __init__(self, config):
        super(Model, self).__init__()

        seed_num = config['seed_num']
        torch.manual_seed(seed_num)
        random.seed(seed_num)
        self.config = config
        self.hidden_size = 50
        self.num_layers = 1
        self.dropout = 0
        self.input_size = 4
        self.output_size = 121

        self.lstm = nn.LSTM(self.input_size, self.hidden_size,
                            dropout=config['dropout'],
                            num_layers=self.num_layers,
                            batch_first=True
                            )

        if config['init_weight']:
            print('Init_ing W ')
            init.xavier_normal_(self.lstm.all_weights[0][0], gain=np.sqrt(config['init_weight_value']))
            init.xavier_normal_(self.lstm.all_weights[0][1], gain=np.sqrt(config['init_weight_value']))

        self.hidden_dim2res = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, inputs):
        inputs = self.dropout(inputs)
        lstm_out, _ = self.lstm(inputs)
        logit = self.hidden_dim2res(lstm_out[:, -1, :])
        return logit


if __name__ == '__main__':
    ne = LSTM(config)
    print(ne)
    test_data = torch.zeros([10, 40, 4])

    test_outs = ne(test_data)
    print(test_outs.shape)
    # [10, 121]
