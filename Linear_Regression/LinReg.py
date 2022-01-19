import numpy as np

class LinearRegression:

    def __init__(self, lr=0.001, n_iters = 1000):

        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y): #We will implement the Gradient Descent Method
        #Initialize the weights
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        #Gradient Descent 
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y)) #We are taking the derivative of the cost function (J) with respect to the weights.
            db = (1/n_samples) * np.sum(y_predicted - y) #We are taking the derivative of the cost function (J) with respect to the bias.
            
            #Learning Step
            self.weights -= self.lr * dw 
            self.bias -= self.lr * db


    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        
        return y_predicted


