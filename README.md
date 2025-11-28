# Used Car Price Prediction

This project predicts the prices of used cars using machine learning models in Python.

---

## Project Overview
The goal of this project is to predict the price of used cars based on simple features such as:

- Model year
- Mileage

We compare the performance of three models to determine the most accurate predictor.

---

## Dataset
- **Source:** `used_cars.csv`
- **Features:**
  - `brand`
  - `model`
  - `model_year`
  - `milage`
  - `price` (target variable)
- **Rows:** 4009 after cleaning

---

## Data Preprocessing
- Converted `price` from string to numeric (removed `$` and `,`)  
- Converted `milage` from string to numeric (removed `,` and `mi.`)  
- Dropped rows with missing `price` values  
- Selected features for prediction: `model_year` and `milage`

---

## Models Used
1. **Linear Regression** – baseline model  
2. **Decision Tree Regressor** – captures non-linear relationships  
3. **Random Forest Regressor** – ensemble method for better accuracy

---

## Model Training
- Split dataset into training (80%) and testing (20%) sets  
- Trained all three models on training data  
- Made predictions on test data

---

## Evaluation Metric
- **Mean Squared Error (MSE)**: measures average squared difference between predicted and actual prices  
- Lower MSE → better model performance

---


## Best Model
- **Random Forest Regressor** performed best (lowest MSE)  
- Reason: Combines multiple trees, reducing overfitting and improving accuracy  

---

## Conclusion
- Predicting used car prices is feasible with simple features and ML models  
- Random Forest gives the most accurate predictions  
- Future improvements: add more features (brand, model type, condition) to increase accuracy

---


