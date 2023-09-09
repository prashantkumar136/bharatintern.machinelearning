import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error
import pandas as pd


iris=datasets.load_iris()

iris_x=iris.data


iris_x_train=iris_x[:-50]
iris_x_test=iris_x[-50:]

iris_y_train=iris.target[:-50]
iris_y_test=iris.target[-50:]

model=linear_model.LinearRegression()

model.fit(iris_x_train,iris_y_train)
iris_y_predict=model.predict(iris_x_test)

print("mean squared error is:   ",mean_squared_error(iris_y_test,iris_y_predict))
print("weigths:  ",model.coef_)
print("intercet: ",model.intercept_)

# plt.scatter(iris_y_test,iris_x_test)
plt.plot(iris_x_test,iris_y_predict)