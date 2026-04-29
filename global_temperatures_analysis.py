#Project must be on python version 3.12 or 3.11
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# read csv file to get info.
df = pd.read_csv("data/GlobalTemperatures.csv")

#Convert dt into datetime
df["dt"] = pd.to_datetime(df["dt"])

#Get Year from datetime
df["Year"] = df["dt"].dt.year

#Remove the Not present values from Land and Ocean average temp as some ocean weren't calculated back then. 
df = df.dropna(subset=["LandAndOceanAverageTemperature"])

#Get the Yearly average of temp instead of each month. 
yearly = df.groupby("Year")["LandAndOceanAverageTemperature"].mean()


#Setting the style for my Seaborn and pyplot graphs on a simple whitegrid to make data more clear.
sns.set_theme(style="whitegrid")
#Sets the width and height of the future graphs.
plt.figure(figsize=(12, 6)) 

#Creating a lineplot of the years and temperature to view the data.
sns.lineplot(x=yearly.index, y=yearly.values, label="Yearly Temperature Average", alpha=0.4)

#Creates a smoothed out rolling line to go in between the graph values to show a more clear trend.
rolling_avg = yearly.rolling(window=10).mean()
sns.lineplot(x=rolling_avg.index, y=rolling_avg.values, label="10-Year Rolling Average")

#Creates the titles and lavels for the graph.
plt.title("Global Land and Ocean Temperature Trend (1750–Present)", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Temperature (°C)")

#Ensures data fits and then shows the graph.
plt.tight_layout()
plt.show()

#Format data to create a supervised model to predict future temperatures.
df["Year"] = df["Year"].astype(int)
X = yearly.index.to_numpy().reshape(-1, 1)
y = yearly.values

#Creating Linear regression model.
model = LinearRegression()
model.fit(X, y)
#Determining how accurate the line of prediction is. 
score = model.score(X, y)
print("Linear Regession Info:")
print(f"Linear R² Score: {score:.4f}")
#predicting Temp given year. It shows that given the data other models may be more accurate. 
y_pred = model.predict(X)

#Creating a new graph to include regression.
plt.figure(figsize=(12,6))

# Actual data
sns.lineplot(x=yearly.index, y=yearly.values, label="Actual Data", alpha=0.4)

# Regression line
sns.lineplot(x=yearly.index, y=y_pred, label="Regression Line")

plt.title("Global Temperature Trend with Regression Line")
plt.xlabel("Year")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.show()

#Predicting future Temps.
future_years = np.array([2030, 2040, 2050]).reshape(-1, 1)
future_preds = model.predict(future_years)
print("Linear Regression future temp predictions for the next three decades:")
for year, temp in zip(future_years.flatten(), future_preds):
    print(f"Linear Prediction - {year}: {temp:.2f} °C")

#This result is not completely accurate and does not allow for a more advanced and realistic view on temp change. To solve this I'll use polynomial regression

#Polynomial regression code:
#Setting data to have X^2 and transforming previous data to fit a polynomial system. 
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
#Still uses linear regression system but now with polynomial features. We train the new model. 
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

#Creating the prediction for the temp given the X_poly. 
y_poly_pred = poly_model.predict(X_poly)

poly_score = poly_model.score(X_poly, y)
print("Polynomial Regression Info:")
print(f"Polynomial R² Score: {poly_score:.4f}")


# Transform future years intpo usable form for Polynomial Regression
future_poly = poly.transform(future_years)

# Predict using polynomial model
future_preds_poly = poly_model.predict(future_poly)

#Print out prediction
print("Polynomial Regression future temp predictions for the next three decades:")
for year, temp in zip(future_years.flatten(), future_preds_poly):
    print(f"Polynomial Prediction - {year}: {temp:.2f} °C")

#Results show around a 10% increase in the R² Score with Polynomial Regression as well as higher overall temps. 

#Create a comparsion Graph between Linear and Polynomial Regression

plt.figure(figsize=(12,6))

# Actual data
sns.lineplot(x=yearly.index, y=yearly.values, label="Actual Data", alpha=0.4)

# Linear model
sns.lineplot(x=yearly.index, y=y_pred, label="Linear Regression")

# Polynomial model
sns.lineplot(x=yearly.index, y=y_poly_pred, label="Polynomial Regression")

plt.title("Model Comparison: Linear vs Polynomial")
plt.xlabel("Year")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.show()

