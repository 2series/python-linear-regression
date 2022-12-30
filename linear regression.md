# Simple Linear Regression Python
## Simple linear regression with evaluating the fitness of the model with a cost function

>Here is a step-by-step guide to implementing simple linear regression with a cost function in Python:

1. Import the necessary libraries (e.g. NumPy, Matplotlib)
2. Load the data and split it into training and test sets
3. Define the cost function, which measures the difference between the predicted values and the true values

The cost function for simple linear regression is given by:

>cost(h(x), y) = 1/2m * sum((h(x) - y)^2)

where h(x) is the predicted value, 
y is the true value, and 
m is the number of rows in the dataset

4. Initialize the model parameters (i.e. the slope and intercept)
5. Implement the gradient descent algorithm to optimize the cost function and find the optimal model parameters

*The gradient descent algorithm works by repeatedly updating the model parameters in the direction that minimizes the cost function*

6. Use the learned model parameters to make predictions on the test set

7. Evaluate the model by computing the mean squared error (MSE) between the predicted values and the true values

Here is sample code that demonstrates how to implement simple linear regression with a cost function in Python:

```
import numpy as np

# Load the data and split it into training and test sets
X = ...  # feature values
y = ...  # target values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize the model parameters
theta = np.zeros(2)  # 2 parameters: slope and intercept

# Define the cost function
def cost_function(X, y, theta):
    m = len(y)
    h = X @ theta  # predicted values
    cost = 1/(2*m) * np.sum((h - y)**2)
    return cost

# Implement the gradient descent algorithm
learning_rate = 0.01
num_iterations = 1000
for i in range(num_iterations):
    h = X_train @ theta  # predicted values
    error = h - y_train  # error between predicted and true values
    gradient = X_train.T @ error  # gradient of the cost function
    theta -= learning_rate * gradient  # update the model parameters

# Make predictions on the test set
y_pred = X_test @ theta

# Evaluate the model
mse = np.mean((y_pred - y_test)**2)
print("Mean Squared Error:", mse)
```

>Here is a step-by-step guide to implementing ordinary least squares (OLS) for simple linear regression in Python:

1. Import the necessary libraries (e.g. NumPy, Matplotlib)
2. Load the data and split it into training and test sets
3. Use the training set to fit the model using OLS 

The OLS solution for simple linear regression is given by:

>theta = (X^T * X)^(-1) * X^T * y

where theta is the vector of model parameters (i.e. the slope and intercept), 
X is the feature matrix, and 
y is the target vector

4. Use the learned model parameters to make predictions on the test set
5. Evaluate the model by computing the mean squared error (MSE) between the predicted values and the true values

Here is sample code that demonstrates how to implement OLS for simple linear regression in Python:

```
import numpy as np

# Load the data and split it into training and test sets
X = ...  # feature values
y = ...  # target values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Add a column of ones to the feature matrix (for the intercept term)
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

# Fit the model using OLS
theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# Make predictions on the test set
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
y_pred = X_test @ theta

# Evaluate the model
mse = np.mean((y_pred - y_test)**2)
print("Mean Squared Error:", mse)
```

>Note that OLS is just one method for fitting a simple linear regression model
Other methods include gradient descent and the normal equation

### Gradient descent: 
This is an iterative optimization algorithm that works by repeatedly updating the model parameters 
in the direction that minimizes the cost function

Here is sample code for implementing gradient descent for simple linear regression in Python:

```
import numpy as np

# Load the data and split it into training and test sets
X = ...  # feature values
y = ...  # target values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize the model parameters
theta = np.zeros(2)  # 2 parameters: slope and intercept

# Define the cost function
def cost_function(X, y, theta):
    m = len(y)
    h = X @ theta  # predicted values
    cost = 1/(2*m) * np.sum((h - y)**2)
    return cost

# Implement the gradient descent algorithm
learning_rate = 0.01
num_iterations = 1000
for i in range(num_iterations):
    h = X_train @ theta  # predicted values
    error = h - y_train  # error between predicted and true values
    gradient = X_train.T @ error  # gradient of the cost function
    theta -= learning_rate * gradient  # update the model parameters

# Make predictions on the test set
y_pred = X_test @ theta

# Evaluate the model
mse = np.mean((y_pred - y_test)**2)
print("Mean Squared Error:", mse)
```

### Normal equation: 
This method directly solves for the optimal model parameters using the closed-form solution given by the normal equation:

>theta = (X^T * X)^(-1) * X^T * y

Here is sample code for implementing the normal equation for simple linear regression in Python:

```
import numpy as np

# Load the data and split it into training and test sets
X = ...  # feature values
y = ...  # target values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Add a column of ones to the feature matrix (for the intercept term)
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

# Fit the model using the normal equation
theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# Make predictions on the test set
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
y_pred = X_test @ theta

# Evaluate the model
mse = np.mean((y_pred - y_test)**2)
print("Mean Squared Error:", mse)
```

## Conclusion
Both **gradient descent** and the normal equation can be used to fit a simple linear regression model, and 
which one to use depends on the specific situation

Gradient descent is generally more flexible and can be used to fit models with a large number of parameters, but 
it may require more computational resources and may not always converge to the optimal solution

On the other hand, the **normal equation** is computationally efficient and always finds the exact optimal solution, but 
it may not scale well to large datasets and may be sensitive to the presence of multicollinearity in the feature matrix

In general, gradient descent is a good choice when the number of examples in the dataset is large, 
while the normal equation is a good choice when the number of examples is small and there is no multicollinearity in the feature matrix


