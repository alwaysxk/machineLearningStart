import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def loadData(path):
	features = []; labels = [];
	cout = 0
	trainFile = open(path); 
	for line in trainFile.readlines():
		currLine = line.strip().split('\t')
		lineArr = []
		lineArr.append(1.0)
		for i in range(21):
			lineArr.append(float(currLine[i]))
		features.append(lineArr)
		labels.append(float(currLine[21]))
		cout += 1
	return features, labels, cout

if __name__ == '__main__':
	train_features, train_labels , cout1= loadData('horseColicTraining.txt')
	test_features, test_lables ,cout2= loadData('horseColicTest.txt')
	clf = LogisticRegression()
	clf.fit(train_features, train_labels)
	print clf.score(test_features, test_lables)