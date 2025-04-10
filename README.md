# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load California housing data, select features and targets, and split into training and testing sets.

2.Scale both X (features) and Y (targets) using StandardScaler.

3.Use SGDRegressor wrapped in MultiOutputRegressor to train on the scaled training data.

4.Predict on test data, inverse transform the results, and calculate the mean squared error.
## Program:
```
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
dataset=fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
print(df.head())
df.info()
x=df.drop(columns=['AveOccup','HousingPrice'])
x.info()
y=df[['AveOccup','HousingPrice']]
y.info()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
scaler_x=StandardScaler()
scaler_y=StandardScaler()
x_train=scaler_x.fit_transform(x_train)
x_test=scaler_x.transform(x_test)
y_train=scaler_y.fit_transform(y_train)
y_test=scaler_y.transform(y_test)
sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(x_train,y_train)
y_pred=multi_output_sgd.predict(x_test)
y_pred=scaler_y.inverse_transform(y_pred)
y_test=scaler_y.inverse_transform(y_test)
mse=mean_squared_error(y_test,y_pred)
print("Mean Squared error : ",mse)
```
# Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
# Developed by:  V AMIRTHAVARSHINI
# RegisterNumber: 212223040014 

## Output:
![multivariate linear regression model for predicting the price of the house and number of occupants in the house](sam.png)
![Screenshot 2025-04-10 112734](https://github.com/user-attachments/assets/7daeb807-6e33-4909-a6d6-41b0f4b179a0)
![Screenshot 2025-04-10 112751](https://github.com/user-attachments/assets/69fecb87-f7d9-47c9-baa4-f5903d2ed75a)
![Screenshot 2025-04-10 112807](https://github.com/user-attachments/assets/abf09017-3ff3-42c5-a1a7-cb3e9c31d356)
![Screenshot 2025-04-10 112830](https://github.com/user-attachments/assets/cbac70a2-d5a2-42c0-a9cc-29f46d517ceb)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
