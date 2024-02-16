# -*- coding: utf-8 -*-
"""
Created on 2024/1/22 

@author: YJC

Purpose：
"""
import numpy
import numpy as np
import matplotlib.pyplot as plt

data_1E_base = np.load('..//datas//dataset//5100_0.1_0.02.npy')
data_1E_rand = np.load('..//datas//dataset//yjc-1-17-rand-chirp.npy')
data_1E_chir = np.load('..//datas//dataset//yjc-1-17-chirp.npy')
data_1E_x_ra = np.load('..//datas//dataset//yjc-1-17-rand-chirp-%10.npy')
data_1E_x_ch = np.load('..//datas//dataset//yjc-1-17-chirp-%10.npy')

data_1 = np.load('..//datas//dataset//yjc-1-17-rand-chirp-%10.npy')

point = 90


def normal_(data: numpy.ndarray):
    move_m, move_s = data[0].mean(), data[0].std()
    forc_m, forc_s = data[1].mean(), data[1].std()
    data[0] = (data[0] - move_m) / move_s
    data[1] = (data[1] - forc_m) / forc_s
    return data


# data_1E_base = normal_(data_1E_base)
# data_1E_rand = normal_(data_1E_rand)
# data_1E_chir = normal_(data_1E_chir)
# data_1E_x_ra = normal_(data_1E_x_ra)
data_1_t = normal_(data_1)

plt.figure(figsize=(16, 8))
plt.title(str(point) + ' move')
# plt.plot(data_1E_base[0][:, point], label='data_1E_2')
plt.plot(data_1_t[0][:, point], label='data_1E_base')
plt.plot(data_1_t[1][:, point], label='data_1E_base')
# plt.plot(data_1E_rand[0][:, point], label='data_1E_rand')
# plt.plot(data_1E_rand[1][:, point], label='data_1E_rand')
# plt.plot(data_1E_x_ch[1][:, point], label='data_1E_x_ch')
plt.legend()

plt.show()

# new_data_1 = data_1E_base[:, 0:3010]
# new_data_2 = data_1E_base[:, 3100:6060]
# new_data_3 = data_1E_base[:, 6200:9150]
# new_data_4 = data_1E_base[:, 9300:12220]

# 1
lens = len(data_1_t[0]) // 4

new_data_1 = data_1[:, 0 * lens:1 * lens]
new_data_2 = data_1[:, 1 * lens:2 * lens]
new_data_3 = data_1[:, 2 * lens:3 * lens]
new_data_4 = data_1[:, 4 * lens:]

plt.figure(figsize=(16, 8))
plt.title(str(point) + ' move')
plt.plot(new_data_1[0][:, point], label='data_1E_rand')
plt.plot(new_data_1[1][:, point], label='data_1E_rand')
plt.legend()

plt.show()

# np.save(f'..//datas//dataset//yjc-1-17-rand-chirp-%10_1.npy', new_data_1)
# np.save(f'..//datas//dataset//yjc-1_2_donya_5100_0.1_1e-4_2.npy', new_data_2)
# np.save(f'..//datas//dataset//yjc-1_2_donya_5100_0.1_1e-4_3.npy', new_data_3)
# np.save(f'..//datas//dataset//yjc-1_2_donya_5100_0.1_1e-4_4.npy', new_data_4)
