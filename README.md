# Task-1

# Name - Suyash Appasaheb Kamalakar
# Code for task 1 supervised ML

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#data stored as numpy array
data = np.array([[21 ,2.5] , [47 ,5.1] , [27 ,3.2] , [75 ,8.5] , [30 ,3.5] , [20 ,1.5] , [88 ,9.2] , [60 ,5.5] , [81 ,8.3] , [25 ,2.7] , [85 ,7.7] , [62 ,5.9] ,
     [41 ,4.5] , [42 ,3.3] , [17 ,1.1] , [95 ,8.9] , [30 ,2.5] , [24 ,1.9] , [67 ,6.1] , [69 ,7.4] , [30 ,2.7] , [54 ,4.8] , [35 ,3.8] , [76 ,6.9] , [86 ,7.8]])

#converting data into pandas dataframe
X = data[: , 1]
Y = data[: , 0]
df = pd.DataFrame()
df['Hours Studied'] = X.tolist()
df['Scores'] = Y.tolist()
print(df)

#Scatter plot of given data
df.plot(x='Hours Studied', y='Scores', style='o')
plt.title('Hours Studied Vs Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.show()

# Applying linear regression
x = df.iloc[:,:-1].values
y = df.iloc[:,1].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1,random_state=0)
regressor = LinearRegression()
regressor.fit(x_train,y_train)
LinearRegression()

# Forming line y = mx + c
L = regressor.coef_*x+regressor.intercept_

# Plotting the best fit line
plt.title("Hours Studied Vs Scores")
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.scatter(x,y,color = 'Blue')
plt.plot(x,L,color ='Black')
plt.show()

y_pred = regressor.predict(x_test)
print(y_pred)

#Comparing values
compare = pd.DataFrame({'True Value':y_test, 'Predicted Value':y_pred})
print(compare)

#Finding Predicted score for 9.25 hours of Studying
h = 9.25
Prediction = regressor.predict([[h]])
print("Hours Studeid =" + str(h))
print("Predicted Score = "+ str((Prediction[0])))
