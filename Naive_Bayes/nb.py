import numpy as np

class NaiveBayes:

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
            self._priors[index] = X_c.shape[0] / float(n_samples) #Frequency: Calculates how often is the class c occuring
    
    
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

