import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create images folder
os.makedirs("images", exist_ok=True)

# Dataset
data = {
    "square_footage": [1000, 1500, 1800, 2400, 3000, 3500, 4000],
    "bedrooms": [2, 3, 3, 4, 4, 5, 5],
    "bathrooms": [1, 2, 2, 3, 3, 4, 4],
    "price": [200000, 300000, 350000, 450000, 550000, 650000, 720000]
}

df = pd.DataFrame(data)

# Features and target
X = df[["square_footage", "bedrooms", "bathrooms"]]
y = df["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Predict new house
new_house = pd.DataFrame({
    "square_footage": [2500],
    "bedrooms": [4],
    "bathrooms": [3]
})

predicted_price = model.predict(new_house)
print("Predicted House Price:", predicted_price[0])

# 1. Square Footage vs Price
plt.figure(figsize=(7, 5))
plt.scatter(df["square_footage"], df["price"])
plt.xlabel("Square Footage")
plt.ylabel("Price")
plt.title("Square Footage vs House Price")
plt.savefig("images/square_footage_vs_price.png")
plt.show()

# 2. Bedrooms vs Price
plt.figure(figsize=(7, 5))
plt.scatter(df["bedrooms"], df["price"])
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.title("Bedrooms vs House Price")
plt.savefig("images/bedrooms_vs_price.png")
plt.show()

# 3. Bathrooms vs Price
plt.figure(figsize=(7, 5))
plt.scatter(df["bathrooms"], df["price"])
plt.xlabel("Bathrooms")
plt.ylabel("Price")
plt.title("Bathrooms vs House Price")
plt.savefig("images/bathrooms_vs_price.png")
plt.show()

# 4. Actual vs Predicted
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.savefig("images/actual_vs_predicted.png")
plt.show()

# 5. Residual Error Plot
errors = y_test - y_pred

plt.figure(figsize=(7, 5))
plt.scatter(y_pred, errors)
plt.axhline(y=0)
plt.xlabel("Predicted Price")
plt.ylabel("Error")
plt.title("Residual Error Plot")
plt.savefig("images/residual_error_plot.png")
plt.show()