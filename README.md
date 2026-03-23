# 📈 Stock Price Prediction — Apple Inc. (AAPL)

> **Task 2 | Data Science Portfolio Project**
> Predict next day's closing stock price using Machine Learning.

---

## 📌 About This Project

This is **Task 2** of my Data Science learning journey.

The goal is to use historical stock market data to predict the
**next day's closing price** of Apple Inc. (AAPL) using two
Machine Learning models and compare their performance.

---

## 🎯 Objective

Use historical stock data (Open, High, Low, Volume) to predict
the next day's closing price using:
- ✅ Linear Regression
- ✅ Random Forest Regressor

---

## 📂 Project Structure
```
stock-price-prediction/
│
├── stock_prediction.py           # Main Python script
├── requirements.txt              # All dependencies
│
├── lr_actual_vs_predicted.png    # Linear Regression plot
├── rf_actual_vs_predicted.png    # Random Forest plot
├── model_comparison.png          # Both models comparison
├── error_distribution.png        # Prediction error analysis
├── feature_importance.png        # Feature importance chart
├── metrics_comparison.png        # MAE, RMSE, R² bar chart
│
└── README.md                     # Project documentation
```

---

## 📊 Dataset

| Property      | Details                              |
|---------------|--------------------------------------|
| Stock         | Apple Inc. (AAPL)                    |
| Source        | Yahoo Finance via yfinance library   |
| Date Range    | Jan 2022 – Nov 2024                  |
| Total Rows    | 756 trading days                     |
| Features Used | Open, High, Low, Volume + engineered |
| Target        | Next Day's Closing Price             |

---

## 🔧 Features Used

| Feature          | Description                      |
|------------------|----------------------------------|
| Open             | Opening price of the day         |
| High             | Highest price of the day         |
| Low              | Lowest price of the day          |
| Volume           | Number of shares traded          |
| Price_Change     | Close minus Open                 |
| High_Low_Range   | High minus Low (daily range)     |
| MA_5             | 5-day moving average             |
| MA_10            | 10-day moving average            |
| Volatility       | 5-day rolling standard deviation |
| Prev_Close       | Previous day's closing price     |

---

## 🤖 Models Used

### 1️⃣ Linear Regression
- Simple and interpretable baseline model
- Features scaled using StandardScaler
- Good for understanding linear relationships

### 2️⃣ Random Forest Regressor
- Ensemble of 200 decision trees
- Handles non-linear relationships well
- Provides feature importance scores

---

## 📈 Model Results

| Metric    | Linear Regression | Random Forest |
|-----------|:-----------------:|:-------------:|
| MAE       | $1.95             | $2.18         |
| RMSE      | $2.44             | $2.65         |
| R² Score  | 0.9048            | 0.8881        |

> ✅ Linear Regression achieved R² of 0.90 meaning it explains
> 90% of the variance in stock prices

---

## 📸 Visualizations

| Plot | Description |
|------|-------------|
| lr_actual_vs_predicted.png | LR predictions vs real prices |
| rf_actual_vs_predicted.png | RF predictions vs real prices |
| model_comparison.png | Side by side model comparison |
| error_distribution.png | How far off predictions were |
| feature_importance.png | Which features matter most |
| metrics_comparison.png | MAE RMSE R² bar charts |

---

## 🛠️ Tech Stack

| Library       | Purpose                             |
|---------------|-------------------------------------|
| yfinance      | Fetch stock data from Yahoo Finance |
| pandas        | Data loading and manipulation       |
| numpy         | Numerical operations                |
| scikit-learn  | ML models and evaluation metrics    |
| matplotlib    | Plotting and visualization          |
| seaborn       | Statistical visualization           |

---

## 🚀 How to Run

**Step 1 — Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/stock-price-prediction.git
cd stock-price-prediction
```

**Step 2 — Install dependencies**
```bash
pip install yfinance pandas numpy scikit-learn matplotlib seaborn
```

**Step 3 — Run the script**
```bash
python stock_prediction.py
```

---

## 💡 Key Insights

- Previous Close and Moving Averages are the
  most important features for prediction
- Both models achieve R² above 0.88
- Linear Regression performs very well for
  short term stock price prediction
- Prediction errors follow a normal distribution
  centered near zero showing no systematic bias

---

## ⚠️ Disclaimer

This project is for educational purposes only.
It is NOT financial advice.
Do not use this model for real trading decisions.

---

## 👤 Author

**Your Name Here**

LinkedIn — your linkedin link here
GitHub — your github link here

---

## 📄 License

MIT License — free to use and adapt for your own portfolio.
