# 🌍 Global Temperature Analysis & Prediction

## 📌 Overview

This project analyzes historical global temperature data and applies machine learning models to understand trends and predict future temperatures.

The goal was to explore climate patterns and compare how different regression models perform on real-world data.

---

## 📊 Dataset

* Source: Kaggle (Global Temperature dataset)
* Features used:
  * Date (`dt`)
  * Land and Ocean Average Temperature

---

## 🧹 Data Processing

* Converted date column to datetime format
* Extracted year from date
* Removed missing values
* Aggregated monthly data into yearly averages

---

## 📈 Visualization

* Line plot of yearly global temperatures
* 10-year rolling average to smooth trends
* Regression comparison graphs

---

## 🤖 Machine Learning Models

### 1. Linear Regression

* Assumes a constant rate of temperature increase
* R² Score: ~0.74
* Provides a general trend but underfits the data

### 2. Polynomial Regression (Degree 2)

* Captures curvature in temperature trends
* R² Score: Higher than linear (~10% improvement)
* Models accelerating temperature changes more effectively

---

## 🔮 Predictions

Predictions were made for:

* 2030
* 2040
* 2050

### Key Insight:

* Linear model predicts steady growth
* Polynomial model predicts accelerating warming

---

## ⚠️ Limitations

* Only uses "Year" as a feature
* Does not account for external factors (CO₂, geography, etc.)
* Polynomial regression may overestimate future values (overfitting risk)

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## 🚀 How to Run

1. Clone the repository
2. Install dependencies:

   ```
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Run the script:

   ```
   python global_temperatures_analysis.py
   ```

---

## 💡 Key Takeaways

* Real-world data is rarely perfectly linear
* Model choice significantly impacts predictions
* Understanding assumptions is as important as coding

---

## 📁 Project Structure

```
Global_Warming/
│
├── data/
│   └── GlobalTemperatures.csv
│
├── global_temperatures_analysis.py
└── README.md
```
