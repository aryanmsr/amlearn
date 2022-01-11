import numpy as np 

class LogisticRegression:

    def __init__(self, lr=0.001, n_iters = 1000):

        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y): 
        #Initialize the weights
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        #Gradient Descent 
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model) #Instead of predicting y with a linear function, we will use a sigmoid function to obtain the probabilities.

            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y)) #We are taking the derivative of the cost function (J) with respect to the weights.
            db = (1/n_samples) * np.sum(y_predicted - y) #We are taking the derivative of the cost function (J) with respect to the bias.
            
            #Learning Step
            self.weights -= self.lr * dw 
            self.bias -= self.lr * db


    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_class

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
