# -*- coding: utf-8 -*-
"""
Created on 2024/2/5 

@author: YJC

Purpose：
"""

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
x = iris.data
y = iris.target
print(x.shape, y.shape)
x = np.zeros([150, 121]) + 1
y[:100] += 2
y = np.zeros([150, 2])
y[:100, 0] += 1
print(x.shape, y.shape)
rbf_svc = SVC(kernel='rbf')
rbf_svc.fit(x, y)
y_pred = rbf_svc.predict(x)
acc = accuracy_score(y, y_pred)

print(acc)