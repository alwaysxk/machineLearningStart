#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors 

def iris_type(s):
	it = {'Iris-setosa' : 0, 'Iris-versicolor':1, 'Iris-virginica': 2}
	return it[s]

iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'

if __name__ == '__main__':
	data = np.loadtxt("iris.data", dtype=float, delimiter=',', converters={4: iris_type})
	x, y = np.split(data, (4, ), axis = 1)	
	x = x[:, :2] # 为了作图，这里只保留了前两个特征

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1) # 划分数据集
	
	# 创建多个不同参数的SVM分类器
	clfs = [svm.SVC(C=0.3, kernel='linear'),
			svm.SVC(C=10, kernel='linear'),
			svm.SVC(C=5, kernel='rbf', gamma=1),
			svm.SVC(C=5, kernel='rbf', gamma=14)]
	
	titles = 'Linear, C=0.3', 'Linear, C=10', 'RBF, gamma=1','RBF,gamma=14' 
	x1_min, x1_max = x[:, 0].min(), x[:,0].max()
	x2_min, x2_max = x[:, 1].min(), x[:,1].max()
	x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]
	grid_test = np.stack((x1.flat, x2.flat), axis=1)

	cm_light = mpl.colors.ListedColormap(['#A0FFA0','#FFA0A0','#A0A0FF'])
	cm_dark = mpl.colors.ListedColormap(['g','r','b'])

	matplotlib.rcParams['font.sans-serif']=[u'SimHei']
	matplotlib.rcParams['axes.unicode_minus']= False
	
	plt.figure(figsize=(10,8),facecolor='w')

	for i, clf in enumerate(clfs): 
		clf.fit(x_train,y_train.ravel()) # 训练
		print '准确率 ： ', clf.score(x_test, y_test)
		# print "numOfSupportVector : ", clf.n_support_

		plt.subplot(2,2,i+1) # 画四个图， 两行两列
		grid_hat = clf.predict(grid_test)
		grid_hat = grid_hat.reshape(x1.shape)
		
		plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light, alpha=0.8) # 对区域进行上色
		plt.scatter(x[:,0], x[:,1], c = y, edgecolors='k', s = 40, cmap=cm_dark) # 画出所有样本
		# 画出支撑向量
		#plt.scatter(x[clf.support_, 0], x[clf.support_, 1], edgecolors='k', facecolors='none', s=100, marker='o') 
		
		# 本想画出划分超平面，没有画出,这个问题先留在这儿
		#z = clf.decision_function(grid_test)
		#z = z.reshape(x1.shape)
		#plt.contour(x1, x2, z, colors=list('krk'), liststyles=['--','-','--'], linewidths=[1,2,1],levels=[-1,0,1])

		plt.xlim(x1_min, x1_max)
		plt.ylim(x2_min, x2_max)
		plt.title(titles[i])
		plt.xlabel(iris_feature[0], fontsize=10)
		plt.ylabel(iris_feature[1], fontsize=10)
		plt.grid()
	plt.suptitle(u'SVM不同参数的分类', fontsize=18)
	plt.tight_layout(2)
	plt.subplots_adjust(top=0.92)
	plt.show()
