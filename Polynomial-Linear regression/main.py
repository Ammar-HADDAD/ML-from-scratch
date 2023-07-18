from Polynomial_Regression import Polynomial_Regressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression


m,iterr = 150,1000

X,Y = make_regression(n_samples = m, n_features=1,noise=2)
Y = Y + Y**2
X = X.reshape((m,1))
Y = Y.reshape((m,1))

clf = Polynomial_Regressor(degree=2,n_iter=iterr)
error_log = clf.fit(X,Y)
pred = clf.predict(X)

plt.scatter(X,Y)
plt.scatter(X,pred,c='r')


plt.plot(range(iterr),error_log,c='g')
