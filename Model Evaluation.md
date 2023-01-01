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


### Cross-validation
A technique used to evaluate the performance of a machine learning model on unseen data

It involves partitioning the data into a training set and a test set, fitting the model to the training set, and then evaluating the model's performance on the test set

There are several types of cross-validation, including:

### 1. K-fold cross-validation: 
In k-fold cross-validation, the data is partitioned into k folds (or "folds"), and 
the model is trained and evaluated k times, with a different fold used as the test set each time 

The final performance score is the average of the k scores

Here is an example of how to perform k-fold cross-validation in Python using the KFold class from the sklearn.model_selection library:
```
from sklearn.model_selection import KFold

# Set the number of folds
k = 10

# Create the KFold object
kfold = KFold(n_splits=k)

# Initialize a list to store the scores
scores = []

# Iterate over the folds
for train_index, test_index in kfold.split(X):
    # Get the training and test data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Fit the model to the training data
    model.fit(X_train, y_train)
    
    # Evaluate the model on the test data
    score = model.score(X_test, y_test)
    
    # Append the score to the list
    scores.append(score)

# Calculate the mean score
mean_score = np.mean(scores)
```
The code creates a KFold object with k folds and then iterates over the folds using the split method

It gets the training and test data for each fold, fits the model to the training data, and 
evaluates the model's performance on the test data using the model's score method 

The scores for each fold are stored in a list and the mean score is calculated at the end

### 2. Stratified k-fold cross-validation
A variation of k-fold cross-validation that is used when the data is imbalanced, i.e., 
when the target variable has a non-uniform distribution

It ensures that the proportions of the target variable in each fold are the same as the proportions in the overall dataset

Here is an example of how to perform stratified k-fold cross-validation in Python using the StratifiedKFold class from the sklearn.model_selection library:
```
from sklearn.model_selection import StratifiedKFold

# Set the number of folds
k = 10

# Create the StratifiedKFold object
skfold = StratifiedKFold(n_splits=k)

# Initialize a list to store the scores
scores = []

# Iterate over the folds
for train_index, test_index in skfold.split(X, y):
    # Get the training and test data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Fit the model to the training data
    model.fit(X_train, y_train)
    
    # Evaluate the model on the test data
    score = model.score(X_test, y_test)
    
    # Append the score to the list
    scores.append(score)

# Calculate the mean score
mean_score = np.mean(scores)
```
The code is similar to the k-fold cross-validation example, but 
it uses the StratifiedKFold class instead of the KFold class

It also passes the target variable y to the split method to ensure that the 
proportions of the target variable in each fold are the same as the proportions in the overall dataset




