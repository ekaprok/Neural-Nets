"""
LogisticRegression.py
CS440/640: Lab-Week5
Miguel Valdez
Ekaterina Prokopeva
"""

import numpy as np 
import matplotlib.pyplot as plt 
import math as math
from pprint import pprint

class LogisticRegression:
        
    def __init__(self, input_dim, output_dim):
        
        """
        Initializes the parameters of the logistic regression classifer to 
        random values.
        
        args:
            input_dim: Number of dimensions of the input data
            output_dim: Number of classes
        """
    
        #weights
        self.theta = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        #bias
        self.bias = np.zeros((1, output_dim)) #[1x2]
        self.bias[0] = 0.01
        
    #--------------------------------------------------------------------------
    
    def sigmoid(self, X):       #logistic function - sigmoid function

        return 1 / (1 + np.exp(-X))
    
    #--------------------------------------------------------------------------
    
    def compute_cost(self,X, y):
        
        """
        Computes the total cost on the dataset.
        
        args:
            X: Data array
            y: Labels corresponding to input data
        
        
        returns:
            cost: average cost per data sample
        """
        
        y = np.reshape(y,(len(y),1))
        p_1 = self.sigmoid(np.dot(X, self.theta) + self.bias) # predicted P of label 1
        log_l = ((-y)*np.log(p_1) - (1-y)*np.log(1-p_1)) # log-likelihood vector
    
        return log_l.mean()

    #--------------------------------------------------------------------------
        
    def fit(self,X,y):
        
        """
        Learns model parameters to fit the data.
        """  
        learning_rate = 0.01
 
        # average cost for the entire data set
        cost = self.compute_cost(X,y)
 
        # generate arrays
        hot_y = np.zeros(shape=(len(y),2))
 
        for i in range(len(hot_y)):
            if y[i] == 1:
                hot_y[i] = np.array([0,1])
            else:
                hot_y[i] = np.array([1,0])
 
        # 0.10 is the threshold mentioned in class
        while cost > .0001:
            softmax_scores = self.sigmoid(X)
 
            weight_gradient = np.dot(np.transpose(X),softmax_scores - hot_y) * learning_rate
            bias_gradient = np.dot(np.ones(shape=(1,len(X))),softmax_scores - hot_y) * learning_rate
 
            self.theta = self.theta - weight_gradient
            self.bias = self.bias - bias_gradient
 
 
            cost = self.compute_cost(X,y)
            
        
    #--------------------------------------------------------------------------
 
    def predict(self,X):
        
        """
        Makes a prediction based on current model parameters.
        
        args:
            X: Data array
            
        returns:
            predictions: array of predicted labels
        """
        
        z = np.dot(X, self.theta) + self.bias
        exp_z = np.exp(z)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        predictions = np.argmax(softmax_scores, axis = 1)

        return predictions
        
    
#--------------------------------------------------------------------------

def plot_decision_boundary(model, X, y):
    
    """
    Function to print the decision boundary given by model.
    
    args:
        model: model, whose parameters are used to plot the decision boundary.
        X: input data
        y: input labels
    """
    
    x1_array, x2_array = np.meshgrid(np.arange(-4, 4, 0.01), np.arange(-4, 4, 0.01))
    grid_coordinates = np.c_[x1_array.ravel(), x2_array.ravel()]
    Z = model.predict(grid_coordinates)
    Z = Z.reshape(x1_array.shape)
    plt.figure()
    plt.contourf(x1_array, x2_array, Z, cmap=plt.cm.bwr)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
    plt.show()
    


############################################################################### 
  
X = np.genfromtxt("DATA/NonLinear/X.csv", delimiter=",")
y = np.genfromtxt("DATA/NonLinear/y.csv", delimiter=",")

input_dim = 2
output_dim = 2
a = LogisticRegression(input_dim, output_dim)

plot_decision_boundary(a, X, y)

a.fit(X,y)

plot_decision_boundary(a ,X,y)