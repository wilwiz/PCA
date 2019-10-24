#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
Name        : demo.py
Author      : WiZ
Created     : 4th June 2019
Version     : 1.0
Modified    : th March 2019 - add some description
Description : some words
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

"""1.数据导入"""
data = sio.loadmat('data.mat')
X = data['X']
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:, 0], X[:, 1])
plt.xlabel('original_x'); plt.ylabel('original_y')
plt.show()


"""2.PCA算法"""
def pca(X):
    X = (X - X.mean()) / X.std()# 归一化处理
    X = np.matrix(X);cov = (X.T * X) / X.shape[0]# 计算协方差矩阵
    U, S, V = np.linalg.svd(cov)# 奇异值分解
    U_reduced = U[:, :1]# 将数据投影到一维空间
    new_X = np.dot(X, U_reduced)# 新数据
    return new_X, U
new_X,U = pca(X)

"""3.降维数据映射回原空间"""
# 由于新数据是一维的，无法在二维空间展示
# 则采用逆变换，将新数据映射回二维空间
def  recover_X(Z, U, k):
    U_reduced = U[:,:k]
    return np.dot(Z, U_reduced.T)
X_recovered =  recover_X(new_X, U, 1)

fig2, ax = plt.subplots(figsize=(12,8))
ax.scatter(list(X_recovered[:, 0]), list(X_recovered[:, 1]))
plt.xlabel('pca_x'); plt.ylabel('pca_y')
plt.show()






# 请注意，第一主成分的投影轴基本上是数据集中的对角线。
# 当我们将数据减少到一个维度时，我们失去了该对角线周围的变化，所以在我们的再现中，一切都沿着该对角线。


