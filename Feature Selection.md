## Multiple Linear Regression Polynomial Regression
### Techniques for feature selection in multiple linear regression:

### Backward elimination: 
In this method, you start with all the features and iteratively remove the least significant ones 
until you are left with a subset of features that are most relevant to the target variable

You can use a statistical test such as the p-value to determine the significance of each feature

### Forward selection: 
In this method, you start with no features and iteratively add the most significant ones 
until you reach the desired number of features 

Again, you can use a statistical test such as the p-value to determine the significance of each feature

### Ridge regression: 
This is a variant of multiple linear regression that adds a regularization term to the cost function 

The regularization term penalizes large coefficients and helps to prevent overfitting

You can use cross-validation to tune the regularization strength and select the most important features

Here is sample code that demonstrates how to use **backward elimination** to select 
the most important features in a multiple linear regression model:

#### Backward Elimination
```
import statsmodels.api as sm

# Add a column of ones to the feature matrix (for the intercept term)
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

# Fit the model using backward elimination
model = sm.OLS(y_train, X_train)
results = model.fit()

# Remove the least significant features until only the most important ones remain
p_values = results.pvalues
num_features = X_train.shape[1]
while True:
    max_p_value = max(p_values)
    if max_p_value > 0.05:  # threshold for significance
        feature_index = np.argmax(p_values)
        X_train = np.delete(X_train, feature_index, axis=1)
        p_values = np.delete(p_values, feature_index)
    else:
        break
```

#### Forward Selection
```
import statsmodels.api as sm
import numpy as np

# Add a column of ones to the feature matrix (for the intercept term)
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

# Initialize the feature set with the intercept term
feature_set = [0]

# Set the significance threshold
threshold = 0.05

# Iterate until all features have been added
while True:
    max_p_value = 0
    best_feature = -1
    
    # Iterate over the remaining features
    for i in range(1, X_train.shape[1]):
        if i not in feature_set:
            # Add the feature to the feature set
            temp_feature_set = feature_set + [i]
            temp_X_train = X_train[:, temp_feature_set]
            
            # Fit the model
            model = sm.OLS(y_train, temp_X_train)
            results = model.fit()
            
            # Get the p-value of the added feature
            p_value = results.pvalues[-1]
            
            # Update the best feature if necessary
            if p_value > max_p_value:
                max_p_value = p_value
                best_feature = i
    
    # Stop if no more features can be added
    if best_feature == -1:
        break
    
    # Add the best feature to the feature set
    feature_set.append(best_feature)
    
    # Stop if the p-value of the added feature is greater than the threshold
    if max_p_value > threshold:
        break

# Print the selected features
print("Selected features:", feature_set)
```

#### Ridge Regression
```
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectFromModel

# Set the regularization strength
alpha = 1

# Create the ridge regression model
model = Ridge(alpha=alpha)

# Create the feature selector
selector = SelectFromModel(model)

# Fit the feature selector to the training data
selector.fit(X_train, y_train)

# Get the selected features
selected_features = selector.get_support()

# Print the selected features
print("Selected features:", selected_features)
```

### Statistical test p-value

The p-value is a statistical measure that is used to assess the significance of a feature 
in a multiple linear regression model

It is calculated based on the null hypothesis that the feature has no effect on the target variable 

A small p-value (typically less than 0.05) indicates that the null hypothesis can be rejected, and 
therefore the feature is considered significant. 

On the other hand, a large p-value (greater than 0.05) indicates that the null hypothesis cannot be rejected, and 
therefore the feature is considered not significant

Here is a step-by-step guide to using the p-value to determine the significance of a feature in a 
multiple linear regression model:

1. Fit the multiple linear regression model using all the features
2. Calculate the p-values for each feature using the model's pvalues attribute
3. Select a significance threshold (e.g. 0.05)

If the p-value for a feature is greater than the significance threshold, 
consider the feature not significant

If the p-value for a feature is less than the significance threshold, consider the feature significant.

Here is an example of how to use the p-value to determine the significance of a 
feature in a multiple linear regression model:

```
import statsmodels.api as sm

# Add a column of ones to the feature matrix (for the intercept term)
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

# Fit the model
model = sm.OLS(y_train, X_train)
results = model.fit()

# Get the p-values for each feature
p_values = results.pvalues

# Select a significance threshold
threshold = 0.05

# Print the features that are significant
for i, p_value in enumerate(p_values):
    if p_value < threshold:
        print("Feature", i, "is significant (p-value =", p_value, ")")
    else:
        print("Feature", i, "is not significant (p-value =", p_value, ")")
```

This code fits a multiple linear regression model using all the features, 
calculates the p-values for each feature, and 
then prints out which features are significant based on the chosen significance threshold

Note that the p-value is just one method for assessing the significance of a feature in a 
multiple linear regression model

There are other methods such as the adjusted r-squared or the Akaike information criterion 
that you can use to determine the significance of a feature

### Statistical test Adjusted r-squared
The adjusted r-squared is a modified version of the r-squared metric that takes into account the number of features 
in the model 

It is used to assess the goodness of fit of a multiple linear regression model and 
to compare models with different numbers of features

The r-squared is defined as the fraction of the variance in the target variable that is explained by the model 

It ranges from 0 to 1, with a value of 1 indicating that the model perfectly fits the data 

However, the r-squared can increase even if the model becomes worse, 
if you increase the number of features in the model. This is known as the "curse of dimensionality"

To address this issue, the adjusted r-squared is defined as:

adjusted r-squared = 1 - (1 - r-squared) * (n - 1) / (n - p - 1)

where n is the number of examples in the dataset and p is the number of features in the model

The adjusted r-squared penalizes models with a large number of features and 
is a more reliable measure of the model's fit

To compute the adjusted r-squared in Python, 
you can use the rsquared_adj attribute of the OLS model from the statsmodels library:

```
import statsmodels.api as sm

# Add a column of ones to the feature matrix (for the intercept term)
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

# Fit the model
model = sm.OLS(y_train, X_train)
results = model.fit()

# Print the adjusted r-squared
print("Adjusted R-squared:", results.rsquared_adj)
```

### Statistical test Akaike information criterion (AIC)
The Akaike information criterion (AIC) is a measure of the relative quality of a multiple linear regression model

It takes into account the model's fit to the data as well as the number of features in the model, and is defined as:

AIC = 2 * p - 2 * log(L)

where p is the number of features in the model and L is the maximum value of the likelihood function for the model 

The AIC is used to compare different models and to select the model that best balances fit and simplicity

To compute the AIC in Python, you can use the aic attribute of the OLS model from the statsmodels library:

```
import statsmodels.api as sm

# Add a column of ones to the feature matrix (for the intercept term)
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

# Fit the model
model = sm.OLS(y_train, X_train)
results = model.fit()

# Print the AIC
print("AIC:", results.aic)
```

This code fits a multiple linear regression model using the training data, and then prints the AIC value

Note that the AIC is just one method for assessing the quality of a multiple linear regression model 

There are other methods such as the Bayesian information criterion (BIC) or 
the mean squared error that you can use to evaluate the model's fit



