
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv("used_cars.csv")

print("First rows of dataset:")
print(df.head())
print("\nNumber of rows:", len(df))


# ------------------------------------------------------------
# 2. BASIC DATA CLEANING
# (convert price & mileage to numbers)
# ------------------------------------------------------------

# Remove $ and commas from price
df["price"] = (
    df["price"]
    .str.replace("$", "")
    .str.replace(",", "")
    .astype(float)
)

# Remove commas + " mi." from mileage
df["milage"] = (
    df["milage"]
    .str.replace(",", "")
    .str.replace(" mi.", "")
    .astype(float)
)

# Drop rows that have no price (target)
df = df.dropna(subset=["price"])

print("\nCleaned dataset summary:")
print(df[["price", "milage", "model_year"]].describe())


# ------------------------------------------------------------
# 3. SELECT SIMPLE FEATURES (X) AND TARGET (y)
# ------------------------------------------------------------
# We will predict car price using:
# - model_year
# - milage

X = df[["model_year", "milage"]]   # features
y = df["price"]                    # target variable


# ------------------------------------------------------------
# 4. TRAIN/TEST SPLIT
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ------------------------------------------------------------
# 5. CHOOSE 3 MODELS (Simple ones)
# ------------------------------------------------------------
model1 = LinearRegression()
model2 = DecisionTreeRegressor()
model3 = RandomForestRegressor()


# ------------------------------------------------------------
# 6. TRAIN MODELS
# ------------------------------------------------------------
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)


# ------------------------------------------------------------
# 7. MAKE PREDICTIONS
# ------------------------------------------------------------
pred1 = model1.predict(X_test)
pred2 = model2.predict(X_test)
pred3 = model3.predict(X_test)


# ------------------------------------------------------------
# 8. EVALUATE MODELS (using MSE)
# ------------------------------------------------------------
mse1 = mean_squared_error(y_test, pred1)
mse2 = mean_squared_error(y_test, pred2)
mse3 = mean_squared_error(y_test, pred3)

print("\n---------------------------")
print("MODEL PERFORMANCE (MSE)")
print("---------------------------")
print("Linear Regression:", mse1)
print("Decision Tree:", mse2)
print("Random Forest:", mse3)


# ------------------------------------------------------------
# 9. PRINT SIMPLE COMPARISON TABLE
# ------------------------------------------------------------
print("\n==============================")
print(" SIMPLE MODEL COMPARISON")
print("==============================")
print("Model              |   MSE")
print("------------------------------")
print(f"Linear Regression  | {mse1}")
print(f"Decision Tree      | {mse2}")
print(f"Random Forest      | {mse3}")
print("==============================")
