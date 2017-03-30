"""
CS440 Spring 2016
Programming Assignment #2
#ekaterina and miguel
"""

import numpy as np
import matplotlib.pyplot as plt


class NeuralNet:    
    def __init__(self, input_dim, output_dim, epsilon, hidden_dim = 0, reg = 0): 
        """
        Initializes the parameters of the neural network to random values
        """
        # Gradient descent params:
        self.epsilon = epsilon     # learning rate
        self.reg = reg             # regularization

        self.Theta1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.bias1 = np.zeros((1, hidden_dim))
        self.Theta2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
        self.bias2 = np.zeros((1, output_dim))
        
    ############################################################################
    
    def sigmoid(self, X):       #logistic function - sigmoid function
        return 1 / (1 + np.exp(-X))
        
    def compute_cost(self,X, y):
        """
        Computes the total loss on the dataset
        """
        num_samples = len(X)
        # Calculate our predictions using forward propagation
        zin = X.dot(self.Theta1) + self.bias1
        s = self.sigmoid(zin) 
        zout = s.dot(self.Theta2) + self.bias2
        exp_z = np.exp(zout)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
        # Calculate the cross-entropy loss
        cross_ent_err = -np.log(softmax_scores[range(num_samples), y])
        data_loss = np.sum(cross_ent_err)
       
        return 1/num_samples * data_loss
    
    ############################################################################
 
    def predict(self,x):
        """
        Makes a prediction based on current model parameters
        """
        zin = x.dot(self.Theta1) + self.bias1
        s = self.sigmoid(zin) 
        zout = s.dot(self.Theta2) + self.bias2
        exp_z = np.exp(zout)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return np.argmax(softmax_scores, axis=1)
        
    ############################################################################
    
    def fit(self,X,y,num_epochs):
        """
        Learns model parameters to fit the data
        """
        for i in range(num_epochs):
            #Forward propagation
            zin = X.dot(self.Theta1) + self.bias1
            s = 1./(1 + np.exp(-zin)) 
            zout = s.dot(self.Theta2) + self.bias2
            exp_z = np.exp(zout)
            softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True) #num_samples X 2 matrix corresponding to all inputs
                    
            #Backpropagation
            beta_outer = softmax_scores
            beta_outer[range(len(X)), y] -= 1
            beta_inner = beta_outer.dot(self.Theta2.T) * (s - np.power(s,2)) 
            dTheta2 = (s.T).dot(beta_outer)
            dbias2 = np.sum(beta_outer, axis=0, keepdims=True)
            dTheta1 = np.dot(X.T, beta_inner)
            dbias1 = np.sum(beta_inner, axis=0)
            
            #Regularization 
            dTheta2 += self.reg * self.Theta2
            dTheta1 += self.reg * self.Theta1
            
            # Update params of Gradient Decend
            self.Theta2 += -epsilon * dTheta2
            self.bias2 += -epsilon * dbias2
            self.Theta1 += -epsilon * dTheta1
            self.bias1 += -epsilon * dbias1
    

############################################################################

def plot_decision_boundary(pred_func):
    """
    Helper function to print the decision boundary given by model
    """
    # min and max
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Grid of points
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    #contour and training examples plotted
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.bwr)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
    plt.show()

############################################################################


data = 'digits'

if data == 'linear':

    X = np.genfromtxt('DATA/Linear/X.csv', delimiter=',')
    y = np.genfromtxt('DATA/Linear/y.csv', delimiter=',')
    y = y.astype(int)
    plt.figure()
    plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.bwr)
    plt.show()

    
elif data == 'nonlinear':

    X = np.genfromtxt('DATA/NonLinear/X.csv', delimiter=',')
    y = np.genfromtxt('DATA/NonLinear/y.csv', delimiter=',')
    y = y.astype(int)
    plt.figure()
    plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.bwr)
    plt.show()
    
elif data == 'digits':

    X = np.genfromtxt('DATA/Digits/X_test.csv', delimiter=',')
    y = np.genfromtxt('DATA/Digits/y_test.csv', delimiter=',')
    y = y.astype(int)
    plt.figure()
    plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.bwr)
    plt.show()

input_dim = 2 # input layer dimensionality : 2
output_dim = 2 # output layer dimensionality : 0 and 1 -- so 2

# Gradient descent params
epsilon = 0.01
reg = 0.00001
num_epochs = 500

# Fit model
# You can modify the number of hidden layers here
NN = NeuralNet(input_dim, output_dim, epsilon, 11, reg) 
NN.fit(X,y,num_epochs)

print("Cost: {0}".format(NN.compute_cost(X,y)))
predictions = NN.predict(X)
correct = 0.0
for i in range(len(predictions)):
    if predictions[i] == y[i]:
        correct += 1.0
correct /= len(predictions)
print("Accuracy: {0}".format(correct))

# Plot the decision boundary
plot_decision_boundary(lambda x: NN.predict(x))


