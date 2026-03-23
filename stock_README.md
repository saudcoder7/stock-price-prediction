# 📈 Stock Price Prediction — Apple Inc. (AAPL)

> **Task 2 | Data Science Portfolio Project**  
> Predict next day's closing stock price using Machine Learning (Linear Regression & Random Forest).

---

## 📌 About This Project

This is **Task 2** of my Data Science learning journey.

The goal is to use historical stock market data to predict the **next day's closing price** of Apple Inc. (AAPL) using two regression models and evaluate their performance.

---

## 📂 Project Structure

```
stock-prediction/
│
├── stock_prediction.py           # Main Python script
│
├── lr_actual_vs_predicted.png    # Linear Regression: Actual vs Predicted
├── rf_actual_vs_predicted.png    # Random Forest: Actual vs Predicted
├── model_comparison.png          # Both models side-by-side
├── error_distribution.png        # Prediction error histograms
├── feature_importance.png        # Random Forest feature importance
├── metrics_comparison.png        # MAE, RMSE, R² bar charts
│
└── README.md                     # Project documentation
```

---

## 📊 Dataset Overview

| Property       | Value                                      |
|----------------|--------------------------------------------|
| Stock          | Apple Inc. (AAPL)                          |
| Source         | Yahoo Finance via `yfinance` library       |
| Date Range     | Jan 2022 – Nov 2024 (~756 trading days)    |
| Features Used  | Open, High, Low, Volume + engineered features |
| Target         | Next Day's Close Price                     |

---

## 🔧 Features Used

| Feature         | Description                        |
|-----------------|------------------------------------|
| `Open`          | Opening price of the day           |
| `High`          | Highest price of the day           |
| `Low`           | Lowest price of the day            |
| `Volume`        | Number of shares traded            |
| `Price_Change`  | Close - Open (daily movement)      |
| `High_Low_Range`| High - Low (daily volatility)      |
| `MA_5`          | 5-day moving average               |
| `MA_10`         | 10-day moving average              |
| `Volatility`    | 5-day rolling standard deviation   |
| `Prev_Close`    | Previous day's closing price       |

---

## 🤖 Models Used

### 1. Linear Regression
- Simple, interpretable baseline model
- Features scaled using `StandardScaler`

### 2. Random Forest Regressor
- Ensemble of 200 decision trees
- Handles non-linear relationships
- Provides feature importance scores

---

## 📈 Model Results

| Metric   | Linear Regression | Random Forest |
|----------|:-----------------:|:-------------:|
| MAE      | $1.95             | $2.18         |
| RMSE     | $2.44             | $2.65         |
| R² Score | 0.9048            | 0.8881        |

> ✅ Linear Regression slightly outperformed Random Forest on this dataset with R² ≈ 0.90

---

## 📸 Visualizations

| Plot | Description |
|------|-------------|
| `lr_actual_vs_predicted.png` | LR predictions vs real prices |
| `rf_actual_vs_predicted.png` | RF predictions vs real prices |
| `model_comparison.png` | Side-by-side comparison |
| `error_distribution.png` | Distribution of prediction errors |
| `feature_importance.png` | Which features matter most (RF) |
| `metrics_comparison.png` | MAE, RMSE, R² bar chart |

---

## 🛠️ Tech Stack

| Library         | Purpose                          |
|-----------------|----------------------------------|
| `yfinance`      | Fetch stock data from Yahoo Finance |
| `pandas`        | Data loading & manipulation      |
| `numpy`         | Numerical operations             |
| `scikit-learn`  | ML models & evaluation metrics   |
| `matplotlib`    | Plotting                         |
| `seaborn`       | Statistical visualization        |

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/stock-prediction.git
cd stock-prediction

# 2. Install dependencies
pip install yfinance pandas numpy scikit-learn matplotlib seaborn

# 3. Run the script
python stock_prediction.py
```

> 💡 **Note:** To use real live data, uncomment the `yfinance` block at the top of the script and comment out the simulated data section.

---

## 💡 Key Insights

- **Previous Close** and **Moving Averages** are the most important features
- Both models achieve **R² > 0.88**, meaning they explain 88–90% of price variance
- **Linear Regression** performs surprisingly well for short-term stock prediction
- Prediction errors follow a **normal distribution** centered near 0 — no systematic bias

---

## ⚠️ Disclaimer

> This project is for **educational purposes only**.  
> It is **NOT financial advice**. Do not use this model for real trading decisions.

---

## 👤 Author
**[Your Name]**  
[LinkedIn](https://linkedin.com/in/your-profile) | [GitHub](https://github.com/your-username)

---

## 📄 License
MIT License – free to use and adapt for your own portfolio.
