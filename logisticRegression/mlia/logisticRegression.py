import numpy as np
import random

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

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def gradAscent(train_features, train_labels):
	featuresMat = np.mat(train_features)
	labelMat = np.mat(train_labels).transpose()
	m, n = np.shape(featuresMat)
	alpha = 0.001
	maxCycles = 500
	weights = np.ones((n, 1))
	for k in range(maxCycles):
		h = sigmoid(featuresMat * weights)
		error = (labelMat - h)
		weights = weights + alpha * featuresMat.transpose() * error
	return weights
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not 
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

if __name__ == '__main__':
	train_features, train_labels , cout1 = loadData('horseColicTraining.txt')
	test_features, test_lables ,cout2 = loadData('horseColicTest.txt')
	weight = gradAscent(train_features, train_labels)
	errorCount = 0.0
	for i in range(cout2):
		prob = int(classifyVector(np.array(test_features[i]) , weight))
		if prob != test_lables[i]:
			errorCount += 1
	print 1 - errorCount/cout2