# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
data = pd.read_csv(url)

# Display the first few rows of the dataset
print(data.head())

# Create a figure with subplots
plt.figure(figsize=(15, 10))

# Subplot 1: Original data visualization
plt.subplot(2, 2, 1)
plt.scatter(data['Hours'], data['Scores'], color='blue', label='Data points')
plt.title('Study Hours vs Scores (Original Data)')
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.legend()

# Splitting the dataset into features (X) and target (y)
X = data[['Hours']]  # Feature
y = data['Scores']   # Target

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")

# Subplot 2: Training data with regression line
plt.subplot(2, 2, 2)
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.plot(X_train, model.predict(X_train), color='red', label='Regression line')
plt.title('Linear Regression: Training Data')
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.legend()

# Subplot 3: Test data with regression line
plt.subplot(2, 2, 3)
plt.scatter(X_test, y_test, color='green', label='Actual test values')
plt.plot(X_test, y_pred, color='red', label='Predicted values')
plt.title('Linear Regression: Test Data')
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.legend()


# Adjust layout and display the plots
plt.tight_layout()
plt.show()