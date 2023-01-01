## Linear Regression Application
To apply linear regression to a dataset in Python, you can use the LinearRegression class 
from the sklearn.linear_model library 

Here is an example of how to fit a linear regression model and make predictions:
```
import numpy as np
from sklearn.linear_model import LinearRegression

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create the LinearRegression object
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)
```
The code first splits the data into a training set and a test set using the train_test_split function 
from the sklearn.model_selection library 

It then creates a LinearRegression object, fits the model to the training data using the fit method, and 
makes predictions on the test data using the predict method

The predictions are stored in the y_pred variable

To evaluate the model's performance, you can use metrics like mean squared error (MSE) or R-squared.

Here is an example of how to calculate the MSE and R-squared:
```
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Calculate the MSE
mse = mean_squared_error(y_test, y_pred)

# Calculate R-squared
r2 = r2_score(y_test, y_pred)
```
The code uses the mean_squared_error and r2_score functions from the 
sklearn.metrics library to calculate the MSE and R-squared, respectively







