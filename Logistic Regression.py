import numpy as np
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

Training_Data = pd.read_excel("data.xlsx")
Expected_Output = np.asarray(Training_Data.output)

Training_Data = Training_Data.drop(['output'], axis=1)
Training_Data = np.asarray(Training_Data)


def hypothesis(X, theta):
	z = np.dot(theta, X.T)
	return 1/(1+np.exp(-(z))) - 0.00000001

def cost(X, y, theta):
	y1 = hypothesis(X, theta)
	#print(y1)
	return -(1/len(X)) * np.sum(y*np.log(y1) + (1-y) * np.log(1-y1))

def gradient_descent(X, y, theta, alpha, epochs):
	m = len(X)
	J = [cost(X, y, theta)]
	for i in range(0, epochs):
		h = hypothesis(X, theta)
		for i in range(0, X.shape[1]):
			theta[i] -= (alpha/m)*np.sum((h-y)*X[:,i])
		J.append(cost(X, y, theta))
	return J, theta

def predict(X, y, theta, alpha, epochs):
	Col1 = np.add(np.zeros((len(X), 1)), 1)
	X = np.append(Col1, X, axis=1)
	J, th = gradient_descent(X, y, theta, alpha, epochs)
	h = hypothesis(X, theta)
	for i in range(len(h)):
		h[i]=1 if h[i]>=0.5 else 0
	y = list(y)
	acc = np.sum([y[i] == h[i] for i in range(len(y))])/len(y)
	return J, acc

'''
#VISUALIZE DATA
data = pd.read_excel("data.xlsx")
r = np.asarray(data.r)
g = np.asarray(data.g)
b = np.asarray(data.b)
out = np.asarray(data.output)

idx1 = np.where(out == 1)
idx0 = np.where(out == 0)

print(idx0)
r1 = r[idx1]
r0 = r[idx0]

g1 = g[idx1]
g0 = g[idx0]

b1 = b[idx1]
b0 = b[idx0]

fig = plt.figure()
axes = plt.axes(projection='3d')

axes.scatter3D(r1,g1,b1,c='red')
axes.scatter3D(r0,g0,b0,c='blue')
plt.show()
'''


#RUN LOGISTIC REGRESSION AND GRAPH COST OVER ITERATION
X = Training_Data
y = Expected_Output
theta = [0.5]*(X.shape[1] + 1)
J, acc = predict(X ,y, theta, 0.0001, 25000)

print(acc)

plt.figure(figsize = (12, 8))
plt.scatter(range(0, len(J)), J)
plt.show()
