# input is all data
import time

# step1 choose the center
# random or training
import torch.optim as optim
# step2 training parameter
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch


class RBF_network(object):
    def __init__(self, hidden_nums, r_w, r_c, r_sigma):
        super().__init__()
        self.h = hidden_nums
        self.w = 0
        self.c = 0
        self.sigma = 0
        self.r = {"w": r_w,
                  "c": r_c,
                  "sigma": r_sigma}
        self.errList = []
        self.epochs = 0
        self.tol = 1.0e-5
        self.X = 0
        self.Y = 0
        self.n_samples = 0
        self.n_features = 0

    def guass(self, sigma, X, ci):
        return np.exp(-np.linalg.norm((X - ci), axis=1) ** 2 / (2 * sigma ** 2))

    def change(self, sigma, X, c):
        newX = np.zeros((self.n_samples, len(c)))
        for i in range(len(c)):
            newX[:, i] = self.guass(sigma[i], X, c[i])
        return newX

    def init(self):
        sigma = np.random.random((self.h, 1))
        c = np.random.random((self.h, self.n_features))
        w = np.random.random((self.h + 1, 1))
        return sigma, c, w

    def addIntercept(self, X):
        return np.hstack((X, np.ones((self.n_samples, 1))))

    def calSSE(self, prey, y):
        return 0.5 * (np.linalg.norm(prey - y)) ** 2

    def L2(self, X, c):
        m, n = np.shape(X)
        newX = np.zeros((m, len(c)))
        for i in range(len(c)):
            newX[:, i] = np.linalg.norm((X - c[i]), axis=1) ** 2
        return newX

    def train(self, X, Y, iters, draw):
        self.X = X
        self.Y = Y.reshape(-1, 1)
        self.n_samples, self.n_features = X.shape
        sigma, c, w = self.init()
        for i in range(iters):
            hi_output = self.change(sigma, X, c)
            yi_input = self.addIntercept(hi_output)
            yi_output = np.dot(yi_input, w)
            error = self.calSSE(yi_output, Y)
            if error < self.tol:
                break
            self.errList.append(error)
            print(i, '--------------', error)

            delta_w = np.dot(yi_input.T, (yi_output - Y))
            w -= self.r['w'] * delta_w / self.n_samples
            delta_sigma = np.divide(np.multiply(np.dot(np.multiply(hi_output, self.L2(X, c)).T,
                                                       (yi_output - Y)), w[:-1]), sigma ** 3)
            sigma -= self.r['sigma'] * delta_sigma / self.n_samples
            delta_c1 = np.divide(w[:-1], sigma ** 2)
            delta_c2 = np.zeros((1, self.n_features))
            for j in range(self.n_samples):
                delta_c2 += (yi_output - Y)[j] * np.dot(hi_output[j], X[j] - c)
            delta_c = np.dot(delta_c1, delta_c2)
            c -= self.r['c'] * delta_c / self.n_samples
            if (draw != 0) and ((i + 1) % draw == 0):
                self.draw_process(X, Y, yi_output)

        self.c = c
        self.w = w
        self.sigma = sigma
        self.epochs = i

    def draw_process(self, x, y, y_prediction):
        lens = len(x)
        x1 = np.linspace(1, lens, lens)[:, np.newaxis]
        # plt.scatter(x1, x, color='g')
        plt.scatter(x1, y)
        plt.plot(x1, y_prediction, color='r')
        plt.show()

    def predict(self, X):
        self.n_samples, self.n_features = X.shape
        hi_output = self.change(self.sigma, X, self.c)
        yi_input = self.addIntercept(hi_output)
        yi_output = np.dot(yi_input, self.w)
        return yi_output


class RBF_cell_torch(nn.Module):
    def __init__(self, hidden_nums, features):
        super().__init__()
        self.n_samples = 0
        self.n_features = features
        self.h = hidden_nums
        self.w = nn.Parameter(torch.randn(self.h + 1, 1), requires_grad=True)
        self.c = nn.Parameter(torch.randn(self.h, self.n_features), requires_grad=True)
        self.sigma = nn.Parameter(torch.randn((self.h, 1)), requires_grad=True)

        self.epochs = 0
        self.X = 0
        self.Y = 0

    def guass(self, sigma, X, ci):
        return torch.exp(-torch.norm((X - ci), dim=1) ** 2 / (2 * sigma ** 2))

    def change(self, sigma, X, c):
        newX = torch.zeros((self.n_samples, len(c)))
        for i in range(len(c)):
            newX[:, i] = self.guass(sigma[i], X, c[i])
        return newX

    def init(self):
        sigma = np.random.random((self.h, 1))
        c = np.random.random((self.h, self.n_features))
        w = np.random.random((self.h + 1, 1))
        return sigma, c, w

    def addIntercept(self, X):
        return torch.cat((X, torch.ones((self.n_samples, 1))), dim=1)

    def calSSE(self, prey, y):
        return 0.5 * (np.linalg.norm(prey - y)) ** 2

    def forward(self, inputs):
        self.n_samples, self.n_features = inputs.shape
        hi_output = self.change(self.sigma, inputs, self.c)
        yi_input = self.addIntercept(hi_output)
        # print(yi_input.shape)
        # print(self.w.shape)
        yi_output = torch.mm(yi_input, self.w)
        return yi_output


if __name__ == "__main__":
    XX = np.linspace(-5, 5, 500)[:, np.newaxis]
    Y = np.multiply(1.1 * (1 - XX + 2 * XX ** 2), np.exp(-0.5 * XX ** 2))
    Y = torch.from_numpy(Y)
    # rbf = RBF_network(50, 0.1, 0.2, 0.1)
    # rbf.train(XX, Y, 2000, 50)
    rbf = RBF_cell_torch(50, 1)
    criterionL1 = nn.SmoothL1Loss()
    optimizerG = optim.Adam(rbf.parameters(), lr=0.1)
    print(XX.shape)
    # for i in range(1000):
    #     y = rbf(XX)
    #     lossL1 = criterionL1(Y, y)
    #     optimizerG.zero_grad()
    #     lossL1.backward()
    #     optimizerG.step()
    #     lossL1viz = lossL1.item()
    #     print('epoch is ', i, '---------', lossL1viz)



