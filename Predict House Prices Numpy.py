
import numpy as np
import matplotlib.pyplot as plt

# 1. Sample data (Size in sqft, Price in $1000s)
X = np.array([650, 785, 1200, 1500, 1800, 2000, 2300])
y = np.array([70, 82, 115, 140, 160, 180, 200])

# Normalize data (important for gradient descent)
X_mean = np.mean(X)
X_std = np.std(X)
X_norm = (X - X_mean) / X_std

# Reshape X for matrix operations
X_norm = X_norm.reshape(-1, 1)
y = y.reshape(-1, 1)

# Add bias term (column of 1s)
X_b = np.c_[np.ones((X_norm.shape[0], 1)), X_norm]

# 2. Initialize parameters (weights)
theta = np.zeros((2, 1))  # [bias, weight]

# 3. Hyperparameters
learning_rate = 0.01
epochs = 1000
m = len(y)  # number of samples

# 4. Training using Gradient Descent
for epoch in range(epochs):
    predictions = X_b @ theta
    error = predictions - y
    gradients = (2/m) * (X_b.T @ error)
    theta -= learning_rate * gradients

# Final weights
print("Trained parameters (theta):")
print(theta)

# 5. Predict for new size
def predict(size_sqft):
    x_norm = (size_sqft - X_mean) / X_std
    x_input = np.array([1, x_norm])  # Add bias term
    return x_input @ theta

predicted_price = predict(1750)
print(f"Predicted price for 1750 sqft: ${predicted_price[0]*1000:.2f}")


# 6. Plotting
plt.scatter(X, y, color='blue', label="Data")
predicted_line = X_b @ theta
plt.plot(X, predicted_line, color='red', label="Model Prediction")
plt.xlabel("Size (sqft)")
plt.ylabel("Price ($1000s)")

plt.title("House Price Prediction using Linear Regression")
plt.grid(True)
plt.show()
