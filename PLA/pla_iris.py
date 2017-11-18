#简单实现PLA算法，并且用于鸢尾花数据集，该数据集在原数据集上截取了一部分只留下了两个类别，
#并且选择数据集的前两个特征进行训练，以便可视化的展示。

#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors 


def sign(x):
    if x > 0:
        return 1
    else :
        return -1

# pla算法核心
def pla(x, y):
    w = np.zeros(3)  # w向量初始化为[0,0,0]，目标函数w0 * x0 + w1 * x1 + w2 * x2（x0 = 1），w0为截据， w1,w2为数据集特征   
    for i in range(1000): # 迭代1000次， 其实对此数据集来说，迭代几百次就已经将数据正确的分开
        for j in range(x.shape[0]):     # 对于数据集中的每一个样本
            if sign(np.sum(x[j]*w)) != y[j]:  # 用当前w来计算w * xj，看当前样本是否分类错误，如果错误就对w进行更新
                w = w + y[j]*x[j]       # 更新
    return w

# 预测
def prediction(x, w):
    re = np.sum((x * w), axis=1) # x是二维的，w是一维的，x的每一行和w进行内积
    for i in range(re.shape[0]): # 如果每一个样本内积的结果>0就是正例，否则是反例
        if re[i] > 0:
            re[i] = 1
        else :
            re[i] = -1 
    return re

def iris_type(s):
    it = {'Iris-setosa' : -1, 'Iris-versicolor':1, 'Iris-virginica': 2}
    return it[s]

iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'

if __name__ == '__main__':
    data = np.loadtxt("iris.data", dtype=float, delimiter=',', converters={4: iris_type})
    x, y = np.split(data, (4, ), axis = 1)  
    x = x[:, :2] # 为了作图，这里只保留了前两个特征
    
    # 下面对x新插入一列1，使得x0 = 1, 对应w0，表示截据
    b = np.ones(x.shape[0])
    x = np.column_stack((b,x))
    
    # 训练
    w = pla(x, y)
    
    # 将下面区域划分为 500行500列, 然后对这250000进行预测，然后对他们进行上色，就相当于对一片区域进行上色，不同颜色的区域代表不同的类别
    x1_min, x1_max = x[:, 1].min(), x[:,1].max()
    x2_min, x2_max = x[:, 2].min(), x[:,2].max()
    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]
    grid_test = np.stack((x1.flat, x2.flat), axis=1)
    
    b = np.ones(grid_test.shape[0])
    grid_test = np.column_stack((b,grid_test))
    
    # 预测
    grid_hat = prediction(grid_test,w)
    grid_hat = grid_hat.reshape(x1.shape)

    cm_light = mpl.colors.ListedColormap(['#A0FFA0','#FFA0A0'])
    cm_dark = mpl.colors.ListedColormap(['g','r'])
    matplotlib.rcParams['font.sans-serif']=[u'SimHei']
    matplotlib.rcParams['axes.unicode_minus']= False
    
    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light) # 对区域进行上色
    plt.scatter(x[:,1], x[:,2], c = y, edgecolors='k', s = 40, cmap=cm_dark) # 画出所有样本

    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.xlabel(iris_feature[0], fontsize=10)
    plt.ylabel(iris_feature[1], fontsize=10)
    plt.grid()
    plt.show()
