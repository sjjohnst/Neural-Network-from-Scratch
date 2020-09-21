''' IMPORTS '''

import numpy as np
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


''' FUNCTIONS '''

def sigmoid(z):
	#Sigmoid used as activation function for 
	#binary classification purposes
	return 1/(1+np.exp(-(z))) - 0.00000001

def derivative_sigmoid(s):
	#Derivative of sigmoid function
	#Input is the output of a sigmoid (can be a vector of outputs)
	#Used in backpropagation algorithm
	return np.multiply(s, (1 - s))

def loss(y, y1):
	'''
	y is expected output from training set
	y1 is predicted output of neural net
	this function computes the binary cross entropy loss
	'''
	cross_entropy_1 = np.sum(-y*np.log(y1))       		#when y = 1
	cross_entropy_2 = np.sum(-(1-y)*np.log(1-y1)) 		#when y = 0 
	return (cross_entropy_1 + cross_entropy_1)/len(y)   #final cost is sum of each case, averaged over number of examples

def compute_delta(y, y1, ):
	'''
	computes the delta value to be used in backpropagation
	y is expected outputs
	y1 is predicted outputs
	'''
	loss_wrt_y1 = (-1/len(y))*np.sum(np.divide((y - y1), np.multiply(y1, (1-y1))))  #partial derivative of Loss wrt y1
	y1_wrt_zout = np.sum(derivative_sigmoid(y1))/len(y)					#partial derivative of y1 wrt zout (derivative of sigmoid)
	# above values averaged over batch size
	return loss_wrt_y1 * y1_wrt_zout


def forward(x, w1, w2, b1, b2):
	z1 = np.dot(w1, x.T) + b1
	a1 = sigmoid(z1)

	z2 = np.dot(w2, a1) + b2
	a2 = sigmoid(z2)

	return z1, a1, z2, a2




''' LOAD DATASETS '''

Total_Data = pd.read_excel("data.xlsx")				# read excel file which contains 500 labelled examples
Output_Data = np.asarray(Total_Data.output)			# set Output_Data to be the 'output' column of excel sheet; convert to NumPy array
Input_Data = Total_Data.drop(['output'], axis=1)	# set Input_Data to be the r,g,b columns of excel sheet
Input_Data = np.asarray(Input_Data)					# convert data to NumPy array

X = Input_Data[0:400:1]		#set X to be the first 400 input examples
Y = Output_Data[0:400:1] 	#similarily set y to be the first 400 ouput examples

X_test = Input_Data[401:501:1]		#create test set from final 100 training examples
Y_test = Output_Data[401:501:1] 

''' NETWORK '''

# Initialize weights and biases randomly
w1 = 0.01 * np.random.randn(2, 3) #weights for layer 1
w2 = 0.1 * np.random.randn(1, 2) #weights for output layer (layer 2)

b1 = 0.1 * np.random.randn(2, 1)  #biases for layer 1
b2 = 0.1 * np.random.randn(1, 1)  #single bias for output layer

#print(w1)
#print(w2)

alpha = 0.01		#learning rate
epochs = 1	#number of training iterations
batch_size = 10

n = int(len(X)/batch_size)

''' TRAIN NERUAL NET '''

for i in range(epochs):


	for j in range(n):

		k = j*batch_size

		x = X[k:k+batch_size:1]
		y = Y[k:k+batch_size:1]

		#forward propogation
		z1, a1, z2, a2 = forward(a1, w1, w2, b1, b2)
		y1 = a2
		

		#backwards propogation
		delta2 = y1 - y
		delta1 = np.dot(w2.T, delta2)*derivative_sigmoid(a1)
		
		grad1 = np.dot(delta2, a1.T)


'''
for i in range(epochs):
	# set up for 10 epochs initially

	for i in range(int(n)):
		#load next training examples
		k = i*batch_size
		j = k+batch_size

		x = X[k:j:1]
		y = Y[k:j:1]

		
		#forward propogation
		z1, a1, z2, a2 = forward(x, w1, w2, b1, b2)
		y1 = a2
		#print(y1)
		
		#Compute Loss and Print
		L = loss(y, y1)
		print(L)

		#backwards propogation
		delta = compute_delta(y, y1)
		z1_wrt_a1 = derivative_sigmoid(z1)


		#weight 7 and 8 (output weights) needed for calculations
		w7 = w2[0][0]
		w8 = w2[0][1]

		#need values from a1, a2 and x, so average their rows beforehand
		a1_av = np.average(a1, axis=0)
		a2_av = np.average(a2, axis=0)
		x_av = np.average(x, axis=0)

		w2[0][0] = w2[0][0] - alpha*delta*(a1_av[0])		#update w7
		w2[0][1] = w2[0][1] - alpha*delta*(a1_av[1])		#update w8

		w1[0][0] = w1[0][0] - alpha*delta*(w7)*x_av[0]		#update w1
		w1[0][1] = w1[0][1] - alpha*delta*(w7)*x_av[1]		#update w3
		w1[0][2] = w1[0][2] - alpha*delta*(w7)*x_av[2]		#update w5

		w1[1][0] = w1[1][0] - alpha*delta*(w8)*x_av[0]		#update w2
		w1[1][1] = w1[1][1] - alpha*delta*(w8)*x_av[1]		#update w4
		w1[1][2] = w1[1][2] - alpha*delta*(w8)*x_av[2]		#update w6
		


z1 = np.dot(w1, X_test.T) + b1
a1 = sigmoid(z1)

z2 = np.dot(w2, a1) + b2
a2 = sigmoid(z2)

y1 = a2
'''	
