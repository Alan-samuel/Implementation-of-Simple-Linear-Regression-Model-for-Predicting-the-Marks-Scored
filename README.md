# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

Name: Alan Samuel Vedanayagam
Reg. no: 212223040012

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries
2. Set variables for assigning dataset values
3. Import Linear Regression from sklearn
4. Assign the points for representing in the graph
5. Predict the regression for marks by using the representation of the graph
6. Compare the graphs and hence we obtained the linear regression for the given datas

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Alan Samuel Vedanayagam
RegisterNumber:  212223040012
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
mse=mean_squared_error(y_test,y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
plt.scatter(x_train,y_train,color='orange')
plt.plot(x_train, regressor.predict(x_train),color='red')
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color='orange')
plt.plot(x_test, regressor.predict(x_test),color='red')
plt.title("Hours vs Scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
## Output:
![simple linear regression model for predicting the marks scored](sam.png)

![Screenshot 2025-02-27 194946](https://github.com/user-attachments/assets/4d67a0b7-cbe9-4b1c-aad1-2ea941029128)
![Screenshot 2025-02-27 195131](https://github.com/user-attachments/assets/e7371a40-581a-4c9f-99f7-349bca2eb597)
![Screenshot 2025-02-27 195157](https://github.com/user-attachments/assets/3dd41c56-b35c-4c2e-888d-3016c42e239b)
![Screenshot 2025-02-27 195225](https://github.com/user-attachments/assets/0b2a71fc-e973-452d-a588-165057fbcb56)
![Screenshot 2025-02-27 195359](https://github.com/user-attachments/assets/51c681ff-52d1-47ee-8566-a8ad3ba024b9)
![Screenshot 2025-02-27 195448](https://github.com/user-attachments/assets/a55cda88-08d9-405a-942d-dfd67e0f7eda)
![Screenshot 2025-02-27 195509](https://github.com/user-attachments/assets/c6e4bdf1-9e16-4ebf-8389-32474a7941c6)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
