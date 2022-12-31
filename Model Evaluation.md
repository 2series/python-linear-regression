## Model Evaluation

Mean squared error (MSE) is a measure of the difference between the predicted values and 
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


