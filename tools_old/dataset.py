# -*- coding: utf-8 -*-
"""
Created on 2024/1/16 

@author: YJC

Purpose：
"""
from torch.utils.data import Dataset


class TransDataset(Dataset):

    def __init__(self, data):
        self.dataset = data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]
