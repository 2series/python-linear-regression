## Model Evaluation

### Mean squared error (MSE)
A measure of the difference between the predicted values and 
the true values in a dataset

It is commonly used as a loss function in regression problems, where the goal is to 
minimize the MSE in order to improve the model's predictions

The MSE is calculated as the average squared difference between the predicted values and the true values

It is defined as:

>MSE = (1/n) * Σ(y_pred_i - y_true_i)^2

where:
- n is the number of samples 
- y_pred_i is the predicted value for the i-th sample
- y_true_i is the true value for the i-th sample

Here is an example of how to calculate the MSE in Python:
```
import numpy as np

# Calculate the MSE
mse = np.mean((y_pred - y_true)**2)
```
The code calculates the MSE by taking the mean of the squared differences between 
the predicted values y_pred and the true values y_true

### Mean absolute error (MAE) 
A measure of the difference between the predicted values and the true values in a dataset 

It is commonly used as a loss function in regression problems, where the goal is to minimize 
the MAE in order to improve the model's predictions

The MAE is calculated as the average absolute difference between the predicted values and 
the true values

It is defined as:

>MAE = (1/n) * Σ|y_pred_i - y_true_i|

where: 
- n is the number of samples
- y_pred_i is the predicted value for the i-th sample 
- y_true_i is the true value for the i-th sample

Here is an example of how to calculate the MAE in Python:
```
import numpy as np

# Calculate the MAE
mae = np.mean(np.abs(y_pred - y_true))
```
The code calculates the MAE by taking the mean of the absolute differences between the 
predicted values y_pred and the true values y_true

### R-squared
#### AKA the coefficient of determination

A statistical measure that indicates the proportion of the variance in the dependent variable 
that is explained by the independent variables in a regression model

It is a common metric for evaluating the fit of a regression model and ranges from 0 to 1, 
with higher values indicating a better fit

R-squared is calculated as:

>R^2 = 1 - (SS_res / SS_tot)

where: 
- SS_res is the sum of squared residuals (the difference between the predicted values and the true values) 
- SS_tot is the total sum of squares (the difference between the true values and the mean of the true values)

Here is an example of how to calculate R-squared in Python:
```
import numpy as np

# Calculate the sum of squared residuals
ss_res = np.sum((y_pred - y_true)**2)

# Calculate the total sum of squares
y_mean = np.mean(y_true)
ss_tot = np.sum((y_true - y_mean)**2)

# Calculate R-squared
r_squared = 1 - (ss_res / ss_tot)
```
The code calculates the sum of squared residuals and the total sum of squares, and 
then uses these values to calculate R-squared





