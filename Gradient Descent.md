## Gradient Descent
An optimization algorithm that is used to find the values of the parameters (also known as weights) 
in a machine learning model that minimize the cost function 

>It is commonly used in supervised learning problems, such as 
>1. linear regression, 
>2. logistic regression, and 
>3. neural networks

The algorithm works by iteratively adjusting the weights in the direction of the negative gradient of the cost function 
with respect to the weights

The gradient points in the direction of the steepest ascent, so taking the negative gradient moves in the 
opposite direction and helps to find the minimum of the cost function

Here is an example of how to implement gradient descent for linear regression in Python:
```
import numpy as np

# Set the learning rate
alpha = 0.01

# Set the number of iterations
n_iter = 1000

# Initialize the weights with random values
w = np.random.randn(X.shape[1])

# Initialize a list to store the cost at each iteration
costs = []

# Iterate over the number of iterations
for i in range(n_iter):
    # Make predictions using the current weights
    y_pred = X.dot(w)
    
    # Calculate the cost
    cost = np.mean((y_pred - y)**2)
    
    # Append the cost to the list
    costs.append(cost)
    
    # Calculate the gradient
    gradient = 2 * X.T.dot(y_pred - y) / X.shape[0]
    
    # Update the weights
    w = w - alpha * gradient
```
The code sets the learning rate (alpha) and the number of iterations (n_iter) 

It initializes the weights with random values and creates a list to store the cost at each iteration 

It then iterates over the number of iterations and makes predictions using the current weights

It calculates the cost and appends it to the list

It then calculates the gradient and updates the weights by subtracting the learning rate times the gradient


### 1. In linear regression, 
the goal is to find the values of the weights that minimize the mean squared error (MSE) 
between the predicted values and the true values

The MSE is defined as:

>MSE = (1/n) * Σ(y_pred_i - y_true_i)^2

where:
- n is the number of samples
- y_pred_i is the predicted value for the i-th sample 
- y_true_i is the true value for the i-th sample

To minimize the MSE using gradient descent, we need to compute the gradient of the MSE with respect to the weights

The gradient is defined as:

>gradient = 2 * X.T.dot(y_pred - y) / X.shape[0]

where:
- X is the feature matrix 
- y_pred is the vector of predicted values
- y is the vector of true values

The gradient points in the direction of the steepest ascent, 
so taking the negative gradient moves in the opposite direction and helps to find the minimum of the MSE

Here is an example of how to implement gradient descent for linear regression in Python:
```
import numpy as np

# Set the learning rate
alpha = 0.01

# Set the number of iterations
n_iter = 1000

# Initialize the weights with random values
w = np.random.randn(X.shape[1])

# Initialize a list to store the cost at each iteration
costs = []

# Iterate over the number of iterations
for i in range(n_iter):
    # Make predictions using the current weights
    y_pred = X.dot(w)
    
    # Calculate the cost
    cost = np.mean((y_pred - y)**2)
    
    # Append the cost to the list
    costs.append(cost)
    
    # Calculate the gradient
    gradient = 2 * X.T.dot(y_pred - y) / X.shape[0]
    
    # Update the weights
    w = w - alpha * gradient

# Print the final weights
print(w)
```
The code is similar to the previous example, but it includes a loop to iterate over the number of iterations 

It makes predictions using the current weights, calculates the cost, and appends it to the list 

It then calculates the gradient and updates the weights. Finally, it prints the final weights


### 2. In Logistic Regression
The goal is to find the values of the weights that maximize the likelihood of the observed data

The likelihood is defined as:

>likelihood = Π(y_i^(y_true_i) * (1 - y_i)^(1 - y_true_i))

where:
- y_i is the predicted probability of the positive class for the i-th sample
- y_true_i is the true class for the i-th sample (0 or 1)
- n is the number of samples

To maximize the likelihood using gradient descent, 
we need to compute the gradient of the log likelihood with respect to the weights

The gradient is defined as:

>gradient = X.T.dot(y_pred - y) / X.shape[0]

where: 
- X is the feature matrix
- y_pred is the vector of predicted probabilities 
- y is the vector of true classes (0 or 1)

The gradient points in the direction of the steepest ascent, 
so taking the negative gradient moves in the opposite direction and helps to find the maximum of the log likelihood

Here is an example of how to implement gradient descent for logistic regression in Python:
```
import numpy as np

# Set the learning rate
alpha = 0.01

# Set the number of iterations
n_iter = 1000

# Initialize the weights with random values
w = np.random.randn(X.shape[1])

# Initialize a list to store the cost at each iteration
costs = []

# Iterate over the number of iterations
for i in range(n_iter):
    # Make predictions using the current weights
    y_pred = 1 / (1 + np.exp(-X.dot(w)))
    
    # Calculate the cost
    cost = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    
    # Append the cost to the list
    costs.append(cost)
    
    # Calculate the gradient
    gradient = X.T.dot(y_pred - y) / X.shape[0]
    
    # Update the weights
    w = w - alpha * gradient

# Print the final weights
print(w)
```
The code is similar to the previous example, but it includes a loop to iterate over the number of iterations

It makes predictions using the current weights, calculates the cost, and 
appends it to the list

It then calculates the gradient and updates the weights. Finally, it prints the final weights

### 3. In Neural networks
The goal is to find the values of the weights that minimize the loss function

The loss function measures the difference between the predicted values and the true values, and 
is used to train the model to make better predictions

To minimize the loss function using gradient descent, we need to compute the gradient of the loss function 
with respect to the weights

The gradient is calculated using the backpropagation algorithm, which involves propagating the error backwards 
through the network and computing the gradients of the loss function with respect to each weight

Here is an example of how to implement gradient descent for a neural network in Python using the PyTorch library:
```
import torch

# Set the learning rate
learning_rate = 0.01

# Set the number of epochs
n_epochs = 10

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Convert the data to tensors
X_train = torch.from_numpy(X_train).float().to(device)
y_train = torch.from_numpy(y_train).long().to(device)
X_test = torch.from_numpy(X_test).float().to(device)
y_test = torch.from_numpy(y_test).long().to(device)

# Define the neural network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(n_inputs, n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, n_classes)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create the neural network
net = Net().to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# Iterate over the number of epochs
for epoch in range(n_epochs):
    # Set the model to training mode
    net.train()
    
    # Zero the gradients
    optimizer.zero_grad()
    
    # Forward pass
    outputs = net(X_train)
  
# Calculate the loss
loss = criterion(outputs, y_train)

# Backward pass
loss.backward()

# Update the weights
optimizer.step()

# Print the loss
print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")

# Set the model to evaluation mode
net.eval()

# Make predictions on the test set
with torch.no_grad():
    y_pred = net(X_test)

# Calculate the accuracy
accuracy = (y_pred.argmax(dim=1) == y_test).float().mean().item()

# Print the accuracy
print(f"Accuracy: {accuracy:.2f}")
``` 
The code calculates the loss using the cross-entropy loss function,
performs a backward pass to compute the gradients, updates the weights using the optimizer, and 
prints the loss

It then sets the model to evaluation mode, makes predictions on the test set, calculates the accuracy, and prints the accuracy










