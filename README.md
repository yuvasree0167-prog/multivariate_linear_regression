# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries and read the house.csv dataset using pandas.

2.Select input features as Size and Bedrooms, and output values as Price and Occupants.

3.Apply StandardScaler to normalize the input data.

4.Create and train two SGDRegressor models:
    First model predicts house price.
    Second model predicts number of occupants.
    
5.Get user input for house size and bedrooms, scale the input, predict price and occupants, then display the results.

## Program:
```
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("house.csv")
data.columns = data.columns.str.strip()
X = data[['Size', 'Bedrooms']].values
y_price = data['Price']
y_occ = data['Occupants']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
price_model = SGDRegressor(max_iter=5000, learning_rate='constant', eta0=0.01)
occ_model = SGDRegressor(max_iter=5000, learning_rate='constant', eta0=0.01)
price_model.fit(X_scaled, y_price)
occ_model.fit(X_scaled, y_occ)
size = float(input("Enter house size: "))
bed = int(input("Enter number of bedrooms: "))
new_data = scaler.transform([[size, bed]])
pred_price = price_model.predict(new_data)
pred_occ = occ_model.predict(new_data)
print("Predicted Price:", pred_price[0])
print("Predicted Occupants:", round(pred_occ[0]))
```
Developed by: YUVASREE S

RegisterNumber: 212225230314 


## Output:
<img width="380" height="96" alt="Screenshot 2026-04-27 091625" src="https://github.com/user-attachments/assets/9009b229-9f8f-47d1-9b86-ebc0cdbf8ac9" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
