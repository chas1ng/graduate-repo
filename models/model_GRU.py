# -*- coding: utf-8 -*-
"""
Created on 2024/1/16 

@author: YJC

Purpose：
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import numpy as np
import random

seed_num = 223
torch.manual_seed(seed_num)
random.seed(seed_num)


class GRU(nn.Module):

    def __init__(self, config):
        super(GRU, self).__init__()
        self.args = config
        self.hidden_dim = config['lstm_hidden_dim']
        self.num_layers = config['lstm_num_layers']

        self.gru = nn.GRU(1, self.hidden_dim, dropout=config['dropout'], num_layers=self.num_layers)
        self.hidden_dim2res = nn.Linear(self.hidden_dim, 1)

        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, inputs):
        lstm_out, _ = self.gru(inputs)
        lstm_out = torch.transpose(lstm_out, 0, 1)

        return lstm_out


if __name__ == '__main__':
    pass
