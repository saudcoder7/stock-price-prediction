# ============================================================
#  Task 2: Predict Future Stock Prices (Short-Term)
#  Stock: Apple Inc. (AAPL)
#  Models: Linear Regression + Random Forest
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load Data ─────────────────────────────────────────────
# NOTE: In a real environment with internet, replace this block with:
#   import yfinance as yf
#   df = yf.download("AAPL", start="2022-01-01", end="2024-12-31")
#   df.reset_index(inplace=True)
#
# For offline use, we simulate realistic AAPL-like stock data:

np.random.seed(42)
n = 756  # ~3 years of trading days

dates = pd.bdate_range(start="2022-01-01", periods=n)
price = 150.0
prices = [price]
for _ in range(n - 1):
    change = np.random.normal(0.0003, 0.015)
    price *= (1 + change)
    prices.append(price)

prices = np.array(prices)
df = pd.DataFrame({
    'Date':   dates,
    'Open':   prices * (1 + np.random.uniform(-0.005, 0.005, n)),
    'High':   prices * (1 + np.random.uniform(0.002, 0.015, n)),
    'Low':    prices * (1 + np.random.uniform(-0.015, -0.002, n)),
    'Close':  prices,
    'Volume': np.random.randint(50_000_000, 120_000_000, n).astype(float),
})

print("=" * 60)
print("   STOCK PRICE PREDICTION — APPLE INC. (AAPL)")
print("=" * 60)

# ── 2. Inspect Data ──────────────────────────────────────────
print(f"\n📐 Shape       : {df.shape}")
print(f"📅 Date Range  : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"\n🔍 First 5 Rows:")
print(df.head())
print(f"\nℹ️  Info:")
df.info()
print(f"\n📊 Descriptive Statistics:")
print(df.describe())

# ── 3. Feature Engineering ───────────────────────────────────
df = df.sort_values('Date').reset_index(drop=True)

# Target: next day's Close price
df['Target'] = df['Close'].shift(-1)

# Extra features
df['Price_Change']   = df['Close'] - df['Open']
df['High_Low_Range'] = df['High'] - df['Low']
df['MA_5']           = df['Close'].rolling(5).mean()
df['MA_10']          = df['Close'].rolling(10).mean()
df['Volatility']     = df['Close'].rolling(5).std()
df['Prev_Close']     = df['Close'].shift(1)

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

FEATURES = ['Open', 'High', 'Low', 'Volume',
            'Price_Change', 'High_Low_Range',
            'MA_5', 'MA_10', 'Volatility', 'Prev_Close']
TARGET = 'Target'

X = df[FEATURES]
y = df[TARGET]

# ── 4. Train / Test Split ────────────────────────────────────
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
dates_test       = df['Date'].iloc[split:]

scaler  = StandardScaler()
Xs_train = scaler.fit_transform(X_train)
Xs_test  = scaler.transform(X_test)

print(f"\n🔀 Training samples : {len(X_train)}")
print(f"🔀 Testing  samples : {len(X_test)}")

# ── 5. Train Models ──────────────────────────────────────────
lr = LinearRegression()
lr.fit(Xs_train, y_train)
lr_pred = lr.predict(Xs_test)

rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# ── 6. Evaluate ──────────────────────────────────────────────
def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"\n{'─'*40}")
    print(f"  {name}")
    print(f"{'─'*40}")
    print(f"  MAE  : ${mae:.2f}")
    print(f"  RMSE : ${rmse:.2f}")
    print(f"  R²   : {r2:.4f}")
    return mae, rmse, r2

print("\n📈 MODEL EVALUATION RESULTS")
lr_metrics = evaluate("Linear Regression", y_test, lr_pred)
rf_metrics = evaluate("Random Forest",     y_test, rf_pred)

# ── 7. Plots ─────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
BLUE   = "#2196F3"
RED    = "#F44336"
GREEN  = "#4CAF50"
ORANGE = "#FF9800"
PURPLE = "#9C27B0"

# ── Plot 1: Actual vs Predicted (Linear Regression) ──────────
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(dates_test, y_test.values, color=BLUE,   lw=1.5, label="Actual Close")
ax.plot(dates_test, lr_pred,       color=RED,    lw=1.2, linestyle='--', label="LR Predicted", alpha=0.85)
ax.set_title("Linear Regression — Actual vs Predicted Close Price (AAPL)", fontsize=13, fontweight='bold')
ax.set_xlabel("Date"); ax.set_ylabel("Price (USD)")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=30)
ax.legend(); fig.tight_layout()
fig.savefig("lr_actual_vs_predicted.png", dpi=150)
plt.close()

# ── Plot 2: Actual vs Predicted (Random Forest) ──────────────
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(dates_test, y_test.values, color=BLUE,   lw=1.5, label="Actual Close")
ax.plot(dates_test, rf_pred,       color=GREEN,  lw=1.2, linestyle='--', label="RF Predicted", alpha=0.85)
ax.set_title("Random Forest — Actual vs Predicted Close Price (AAPL)", fontsize=13, fontweight='bold')
ax.set_xlabel("Date"); ax.set_ylabel("Price (USD)")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=30)
ax.legend(); fig.tight_layout()
fig.savefig("rf_actual_vs_predicted.png", dpi=150)
plt.close()

# ── Plot 3: Both Models Side-by-Side ─────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
for ax, pred, color, label in zip(
        axes,
        [lr_pred, rf_pred],
        [RED, GREEN],
        ["Linear Regression", "Random Forest"]):
    ax.plot(dates_test, y_test.values, color=BLUE,  lw=1.5, label="Actual")
    ax.plot(dates_test, pred,          color=color, lw=1.2, linestyle='--', label=label, alpha=0.85)
    ax.set_ylabel("Price (USD)")
    ax.legend(); ax.grid(True, alpha=0.3)
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=30)
fig.suptitle("AAPL Stock Price Prediction — Model Comparison", fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig("model_comparison.png", dpi=150)
plt.close()

# ── Plot 4: Prediction Error Distribution ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, pred, color, label in zip(
        axes,
        [lr_pred, rf_pred],
        [RED, GREEN],
        ["Linear Regression", "Random Forest"]):
    errors = y_test.values - pred
    ax.hist(errors, bins=30, color=color, alpha=0.7, edgecolor='white')
    ax.axvline(0, color='black', lw=1.5, linestyle='--')
    ax.set_title(f"{label} — Error Distribution", fontweight='bold')
    ax.set_xlabel("Prediction Error ($)")
    ax.set_ylabel("Frequency")
fig.suptitle("Prediction Error Analysis", fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig("error_distribution.png", dpi=150)
plt.close()

# ── Plot 5: Feature Importance (Random Forest) ───────────────
fig, ax = plt.subplots(figsize=(8, 5))
importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importances)))
importances.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
ax.set_title("Random Forest — Feature Importance", fontsize=13, fontweight='bold')
ax.set_xlabel("Importance Score")
fig.tight_layout()
fig.savefig("feature_importance.png", dpi=150)
plt.close()

# ── Plot 6: Model Metrics Comparison Bar Chart ───────────────
fig, axes = plt.subplots(1, 3, figsize=(11, 4))
metrics   = ['MAE ($)', 'RMSE ($)', 'R² Score']
lr_vals   = list(lr_metrics)
rf_vals   = list(rf_metrics)
bar_colors = [[RED, GREEN]] * 3

for ax, metric, lv, rv in zip(axes, metrics, lr_vals, rf_vals):
    bars = ax.bar(['Linear\nRegression', 'Random\nForest'], [lv, rv],
                  color=[RED, GREEN], edgecolor='white', width=0.5)
    ax.set_title(metric, fontweight='bold')
    for bar, val in zip(bars, [lv, rv]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.95,
                f'{val:.3f}', ha='center', va='top', fontsize=10,
                color='white', fontweight='bold')

fig.suptitle("Model Performance Comparison", fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig("metrics_comparison.png", dpi=150)
plt.close()

print("\n✅ All plots saved:")
print("   lr_actual_vs_predicted.png")
print("   rf_actual_vs_predicted.png")
print("   model_comparison.png")
print("   error_distribution.png")
print("   feature_importance.png")
print("   metrics_comparison.png")
print("=" * 60)
