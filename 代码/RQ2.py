#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
Name        : pca_for_DP.py 
Author      : WiZ
Created     : 2019/10/23 14:19
Version     : 1.0
Description : some words
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")
with warnings.catch_warnings():warnings.simplefilter("ignore")
with warnings.catch_warnings():warnings.filterwarnings("ignore",category=DeprecationWarning)
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier #NNET
from sklearn.neighbors import KNeighborsClassifier # knn
from sklearn.ensemble import RandomForestClassifier #RF
from sklearn import svm #SVM
import time
import pandas as pd
from pandas import Series


def read_folder():
    '''
    读取文件夹中文本的名字
    :return: 文本list
    '''
    import os
    file_path = './data'
    file_list = os.listdir(file_path)
    return file_list

def read_feature_label(filename):
    '''
    给定文本名称，返回特征和标签
    :param filename: 文本名称
    :return: 特征list和标签list
    '''
    filename = filename
    s = pd.read_csv("./data/"+filename, header=None)
    lens =len(s.iloc[0,:])
    labels = s.iloc[:,lens -1]

    feas = s.iloc[0, :lens - 1]
    ser = Series(feas)
    arr = ser.as_matrix()

    for i in range(1, len(s)):
        feas = s.iloc[i, :lens - 1]
        ser = Series(feas)
        temp = ser.as_matrix()
        arr = np.vstack((arr, temp))
    Xs = arr
    buglist = ['false\r','buggy\n','buggy','False','yes','Y','yes\n','Y\n','false\n','FALSE\n','FALSE','False',False,'0']
    notbuglist = ['true\r','clean\n','clean','True','no\n','N','N\n','true\n','TRUE\n','TRUE','True',True,]
    bugnum = 0
    notbugnum = 0
    for bug in labels:
        if bug in buglist:
            bugnum += 1
        elif bug in notbuglist:
            notbugnum += 1
        else:
            notbugnum += 1
    arrs = []
    if bugnum > notbugnum:
        for bug in labels:
            if bug in buglist:
                arrs.append(0)
            elif bug in notbuglist:
                arrs.append(1)
            else:
                arrs.append(1)
    else:
        for bug in labels:
            if bug in buglist:
                arrs.append(1)
            elif bug in notbuglist:
                arrs.append(0)
            else:
                arrs.append(0)
    Ys = np.array(arrs)


    return Xs,Ys

def read(filename, dcp_name,n_component):

    X, y = read_feature_label(filename)

    # =======降维====
    dcp = decomposition(dcp_name,n_component)
    X = dcp.fit_transform(X)


    from sklearn.preprocessing import scale
    X = scale(X)
    y = y

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=2 / 3)
    return x_train, x_test, y_train, y_test

def clf(index,x_train,y_train):
    if index == 1: clf = DecisionTreeClassifier()
    if index == 2: clf = MLPClassifier()
    if index == 3: clf = KNeighborsClassifier()
    if index == 4: clf = RandomForestClassifier()
    if index == 5: clf = svm.SVC(gamma='scale', probability=True)
    return clf

def metric(y_test,y_pred):
    '''
    返回预测结果
    :param y_test: 预测标签
    :param y_pred: 测试标签
    :return: 召回率，f1,auc
    '''
    from sklearn.metrics import recall_score
    from sklearn.metrics import confusion_matrix
    pd = recall_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import f1_score
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return pd, f1, auc

def decomposition(dcp_name,n_component):
    from sklearn.decomposition import PCA  # 主成分分析
    from sklearn.decomposition import KernelPCA  # 核PCA
    from sklearn.decomposition import FactorAnalysis  # 因子分析
    from sklearn.decomposition import FastICA  # 独立成分分析

    if dcp_name == 'pca': dcp = PCA(n_components=n_component)
    if dcp_name == 'kpca': dcp = KernelPCA(n_components=n_component)
    if dcp_name == 'fa': dcp = FactorAnalysis(n_components=n_component)
    if dcp_name == 'ica': dcp = FastICA(n_components=n_component)

    return dcp

def draw(data,plot_title):
    '''
    根据给定数据画箱线图
    :param data:
    :param plot_title: 图像标题
    :return:
    '''
    tang_array = [i for i in data]
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6))

    bplot = plt.boxplot(tang_array,
                        notch=False,  # notch shape
                        vert=True,  # vertical box aligmnent
                        patch_artist=True)  # fill with color

    colors = ['orange',
              'green',
              'lightblue',
             'red',]
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    plt.xticks([y + 1 for y in range(len(tang_array))], ['J48','NN','KNN','RF','SVM'])
    plt.xlabel('Different Classifiers')
    t = plt.title(plot_title)
    plt.show()
    return 0

def mean(data):
    '''
    返回数据的平均值
    :param data:
    :return:
    '''
    res = []
    for i in data:
        i_ = np.array(i)
        res.append(np.mean(i_))
    return res

if __name__ == '__main__':

    file_list = read_folder()
    file_list = sorted(file_list)

    clf_names = [1,2,3,4,5]
    dcp_name = 'pca'

    all_recall = []; all_f1 = []; all_auc = []

    for n_component in range(14,15):
        print("=================运行中，主元数量：",n_component)


        for clf_name in clf_names:
            recall_list = [];
            f1_list = [];
            auc_list = []
            print(clf_name)

            if clf_name == 1: c = 'dt'
            if clf_name == 2: c = 'nnet'
            if clf_name == 3: c = 'knn'
            if clf_name == 4: c = 'rf'
            if clf_name == 5: c = 'svm'
            for filename in file_list:
                x_train, x_test, y_train, y_test = read(filename, dcp_name,n_component)
                clf2 = clf(clf_name, x_train, y_train)
                clf2.fit(x_train, y_train)
                y_pred = clf2.predict(x_test)
                recall, f1, auc = metric(y_test, y_pred)
                recall_list.append(recall); f1_list.append(f1); auc_list.append(auc)

            all_recall.append(recall_list); all_f1.append(f1_list); all_auc.append(auc_list)

    draw(all_recall,'Recall')
    draw(all_f1,'F1-measure')
    draw(all_auc,'AUC')
    a=mean(all_recall)
    b = mean(all_f1)
    c=mean(all_auc)
    print('recall')
    for i in a:print(i)
    print('f1')
    for i in b:print(i)
    print('auc')
    for i in c:print(i)