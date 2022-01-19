import numpy as np

class NaiveBayes:

    """ 
    Overall Goal: Calculate P(y|X) = (P(X|y) * P(y)) / (P(X)), where X is the feature vector (x1,x2,...,xn).
    Since we assume all features are mutually independent (hence the "Naive" in Naive Bayes), 
    we will have to calculate P(y|X) = (P(x1|y) * P(x2|y) * ... * P(xn|y) * P(y)) / (P(X)).
    We will then select the class with the highest probability. Mathematically,
    y = argmax_yP(y|X) = argmax_y(P(x1|y) * P(x2|y) * ... * P(xn|y) * P(y)) / (P(X)). This is equivalent to
    y = argmax_yP(x1|y) * P(x2|y) * ... * P(xn|y) * P(y).
    Taking the logarithm, we end up calculating 
    y = argmax_ylog((P(x1|y)) + log(P(x2|y)) + ... + log(P(xn|y)) + log(P(y)),
    where P(y) is our prior probability (the frequency by which the class occurs) and P(x_i|y) is the class conditional
    probability (probability of observing the data given the class) which is modelled by the Gaussian distribution 
    P(x_i|y) = (1/sqrt(2*pi*var)) * exp(-(x-mean)^2)/2 * var).
    
    """

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y) #This will find the unique elements of the array 
        n_classes = len(self._classes)

        #Let's initialize the mean, variance, and priors.

        self._mean = np.zeros((n_classes, n_features), dtype = np.float64)
        self._var = np.zeros((n_classes, n_features), dtype = np.float64)
        self._priors = np.zeros(n_classes, dtype = np.float64)

        for index, c in enumerate(self._classes):
            X_c = X[c == y]
            self._mean[index,:] = X_c.mean(axis=0)
            self._var[index,:] = X_c.var(axis=0)
            self._priors[index] = X_c.shape[0] / float(n_samples) #Frequency: Calculates how often is the class c occuring. This is P(y) (the prior probability)
    
    
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self, x):

        posteriors = []
        #Let's calculate the Posterior Probability for each class
        for index, c in enumerate(self._classes):
            prior = np.log(self._priors[index])
            class_conditional = np.sum(np.log(self._gaussian(index, x))) 
            posterior = prior + class_conditional
            posteriors.append(posterior)

        # return class with highest posterior probability
        return self._classes[np.argmax(posteriors)]
    
    def _gaussian(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

