Regression models (both linear and non-linear) are used for predicting a real value. 
If your independent variable is time, then you are forecasting future values, otherwise your model is predicting present but unknown values. 

Machine Learning Regression models:
### - Simple Linear Regression
- Training the Simple Linear Regression model on the Training set
```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```
- Predicting the Test set results
 ```python
 y_pred = regressor.predict(X_test)
 ```
 - Visualising
```python
plt.scatter(X_train,y_train, color='red')
plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience (training set)')
plt.xlabel('Years of experience')
plt.ylabel('salary')
```
```python
plt.scatter(X_test,y_test, color='red')
plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience (training set)')
plt.xlabel('Years of experience')
plt.ylabel('salary')
```
- Get the equation
```python
print(regressor.coef_)
print(regressor.intercept_)
```

The ordinary least squares (OLS) method can be defined as a linear regression technique that is used to estimate the unknown         parameters in a model. The method relies on minimizing the sum of squared residuals between the actual (observed values of the        dependent variable) and predicted values from the model.

### - Multiple Linear Regression


### - Polynomial Regression
### - Support Vector for Regression (SVR)
### - Decision Tree Regression
### - Random Forest Regression
