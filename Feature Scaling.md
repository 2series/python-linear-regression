## Standardization: 
A technique that is often used in machine learning to transform variables 
so that they have a mean of 0 and a standard deviation of 1 

This can be useful for algorithms that assume that the input features are standard normally distributed or 
that weight input features equally

To standardize a variable in Python, you can use the StandardScaler class from 
the sklearn.preprocessing library:

```
from sklearn.preprocessing import StandardScaler

# Create the StandardScaler object
scaler = StandardScaler()

# Fit the StandardScaler object to the data
scaler.fit(X)

# Transform the data
X_standardized = scaler.transform(X)
```
This code creates a StandardScaler object and fits it to the data X 

It then uses the transform method to standardize the data

The resulting standardized data is stored in X_standardized

Note that you should only fit the StandardScaler object to the training data, and 
then use it to transform both the training and test data 

This is because the StandardScaler object estimates the mean and standard deviation of the data 
from the training set, and then uses these estimates to standardize the test data


## Normalization 
A technique that is used to scale the values of a variable to a specific range, such as 0 to 1 or -1 to 1 

This can be useful for algorithms that assume that the input features are within a specific range or 
that weight input features differently based on their range

There are several ways to normalize a variable in Python, 
including min-max normalization and z-score normalization

Here is an example of how to perform min-max normalization:
```
# Perform min-max normalization
X_normalized = (X - X.min()) / (X.max() - X.min())
```

The code subtracts the minimum value of X from each element in X, and 
then divides the result by the range (maximum value minus minimum value) of X

The resulting normalized data is stored in X_normalized

Here is an example of how to perform z-score normalization:
```
# Perform z-score normalization
X_normalized = (X - X.mean()) / X.std()
```

This code subtracts the mean of X from each element in X, and 
then divides the result by the standard deviation of X. 

The resulting normalized data is stored in X_normalized

Note that you should only normalize the training data, and
then use the same normalization parameters to normalize the test data

This is because the normalization parameters (e.g., the minimum and maximum values in min-max normalization) 
are estimated from the training set, and then used to normalize the test data









