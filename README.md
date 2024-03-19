# random-notes
random notes on bias, variance, overfitting, underfitting, good fit, ridge and lasso regression

#bias occurs when an algorithm has limited flexibility to learn from data
#such models pay very little attention to training data and oversimplify model therefore validation error or prediction error and training error follow similar trends
#such models always lead to high error on training and test data

#variance defines glorithms sensitivity to specific sets of data
#a model with high variance pays a lot of attention to to training data and does not generalize therefore the validation error or prediction error are far apart from each other
#such models usually perform well on training data but have high error rates on test data

#overfitting: where ml model tries to learn from details along with noise in data and tries to fit each data point on the curve
#as the model has very little flexibility, it fails to predict new data points, and thus the model rejects every new data point during prediction
#reasons for overfitting: data for training is not cleaned and contains noise(garbage values), model has high variance, size of training data used is not enough, model is too complex

#underfitting: where ml model can neither learn relationship between variables in data nor predict or classify a new datapoint
#as model doesn't fully learn the patterns, it accepts every new data point during prediction
#reasons: not cleaned and contains noise, model has high bias,size of training data used not enough, model is too simple

#good fit: line or curve that best fits data neither overfitting nor underfitting models but just right
#regularization: calibrate linear regression models in order to minimize adjusted loss function and prevent over or underfitting

#ridge regression: modifies over or underfitted models by adding penalty equiv to sum of squares of magnitude of coefficients
#use when have many variables re smaller data smaples, model does not encourage convergence towards zero but likely to make them closer to 0 and prevent overfitting
#lasso regression: modifies over or underfitted models by adding penalty equiv to sum of absolute values of coefficients
#use when fitting linear model with fewer variables

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import pandas as pd

# URL of the Boston housing dataset
boston_dataset_url = "http://lib.stat.cmu.edu/datasets/boston"

# Load the dataset into a DataFrame
boston_pd = pd.read_csv(boston_dataset_url)

# Manually specify the column names based on the dataset description
column_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
    'PTRATIO', 'B', 'LSTAT', 'MEDV'
]

# Assign the column names to the DataFrame
boston_pd.columns = column_names

# Display the DataFrame
print(boston_pd.head())

raw_df.columns = raw_df.columns
boston_pd=pd.DataFrame(raw_df.data)
boston_pd.columns=raw_df.feature_names

#apply multiple linear regression model
lreg=LinearRegression()
lreg.fit(x_train,y_train)

#prediction on test
lreg_y_pred=lreg.predict(x_test)

#mse
mean_squared_error=np.mean((lreg_y_pred-y_test)**2)
print("Mean squared error on test set:",mean_squared_error)


#putting toether coefficient and t heir corresponding variable names
lreg_coefficient=pd.Datafram()
lreg_coefficient["Columns"]=x_train.columns
lreg_coefficient["Coefficient Estimate"]=pd.Series(lreg.coef_)
print(lreg_coefficient)

from sklearn.linear_model import Ridge

#train model
ridgeR = Ridge(alpha=1)
ridgeR.fit(x_train,y_train)
y_pred=ridgeR.predict(x_test)

#calculate mean square error
mean_squared_error_ridge=np.mean((y_pred-y_test)**2)
print("Mean Squared error on test set:" , mean_squared_error_ridge)

#get ridge coefficient and print them
ridge_coefficient=pd.DataFram()
ridge_coefficient["Columns"]=x_train.columns
ridge_coefficient["Coefficient Estimate"]=pd.Series(ridgeR.coef_)
print(ridge_coefficient)
