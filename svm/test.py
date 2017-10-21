#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = np.loadtxt("testSet.txt", dtype=float) # 使用numpy库函数读入数据集
    x, y = np.split(data, (2,), axis = 1) # 划分特征与标签, x为data前两列，y为data最后列
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.7)
    clf = svm.SVC(kernel='linear', random_state=1) # svm线性分类器
    clf.fit(x_train, y_train.ravel()) # 训练
    print clf.score(x_test, y_test) # 准确率
        
    # 对区域进行预测，涂色
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]
    # 从x1_min,x2_min, x1_max, x2_max这四个点组成的矩阵划分成 500 * 500 个点，
    # 上面得到x1, x2都是500*500的二维矩阵，x1[i][j]与x2[i][j]可以确定一个点，每个点代表一个测试值(样本)，
    # 而以前x_train或x_test都是两列，下面一行代码将x1,x2合并成两列的矩阵，第一列对应第一个特征值，也就是x1
    grid_test = np.stack((x1.flat, x2.flat), axis = 1)
    # 预测
    grid_hat = clf.predict(grid_test)
    # 这里预测后的结果只有一列，无法与500*500的x1和x2一一对应，所以还要将grid_hat在变成500*500
    # 之后，x1[i][j]对应特征1, x2[i][j]对应特征2，grid_hat[i][j]对应着输出。之后就可以画图了 
    grid_hat = grid_hat.reshape(x1.shape)
    
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r'])
    # 对整个区域上色
    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
    # 画出整个数据集的各样本的分布
    plt.scatter(x[:, 0], x[:, 1], c = y, edgecolors='k', s=50, cmap=cm_dark)
    z = clf.decision_function(grid_test) # 测试样本到划分超平面的距离
    z = z.reshape(x1.shape) 
    # 下面画出到超平面距离为-1，1, 0的直线，分别对应支持向量所在的直线，划分超平面
    plt.contour(x1, x2, z, colors=list('krk'), linestyles=['--', '-', '--'], linewidths=[1, 2, 1], levels=[-1, 0, 1])
    plt.xlabel('x', fontsize=13)
    plt.ylabel('y', fontsize=13)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid()
    plt.show()