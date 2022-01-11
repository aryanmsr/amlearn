import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 
from LogReg import LogisticRegression

breast_cancer = datasets.load_breast_cancer()

X, y = breast_cancer.data, breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1234)

regressor = LogisticRegression(lr=0.0001)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

#Testing the Model

def accuracy(y_true, y_predicted):
    return np.sum(y_true == y_pred) / len(y_true)

accuracy_value = accuracy(y_test, y_pred)
print(accuracy_value)