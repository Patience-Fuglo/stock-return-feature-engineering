# ⚙️ End-to-End Feature Engineering Pipeline: Stock Return Prediction

A complete ML feature engineering pipeline applied to AAPL stock data — building financial features from raw OHLCV data, applying three distribution transformations, implementing three encoding strategies, comparing three scaling methods, and assembling a production-ready `sklearn` Pipeline that trains a Logistic Regression classifier for next-day return direction prediction.

---

## 🧩 Project Overview

Raw stock price data is not a feature set — it's a starting point. This project builds the entire transformation layer between raw OHLCV data and a trained model: momentum indicators, volatility windows, range features, volume signals, distribution corrections, categorical time encodings, and a full `ColumnTransformer` pipeline with imputation, scaling, and one-hot encoding baked in.

The model itself (Logistic Regression, 52% accuracy on held-out data) is deliberately simple. The point is not to achieve maximum predictive performance — it's to build the infrastructure correctly, in a way that would survive production deployment without data leakage.

---

## 🎯 What It Builds

| Stage | Content |
|---|---|
| **1. Feature Creation** | Returns, log returns, 5 & 10-day momentum, 10 & 20-day SMA, 10 & 20-day rolling volatility, daily range (H-L), volume change |
| **2. Feature Transformation** | Log transformation, Box-Cox transformation (λ = 1.007), skewness audit |
| **3. Domain-Driven Design** | Financial intuition guide for each feature — why it exists, what it captures |
| **4. Target Variable** | Binary: next-day return direction (1 = up, 0 = down) via `shift(-1)` |
| **5. Encoding** | One-hot encoding, label encoding, target encoding — for day-of-week and month-name calendar features |
| **6. Scaling** | StandardScaler, MinMaxScaler, RobustScaler — side-by-side comparison |
| **7. Pipeline Assembly** | `sklearn Pipeline` + `ColumnTransformer` with imputation → scaling → OHE |
| **8. Model & Evaluation** | Logistic Regression; accuracy, precision, recall, F1 on 20% hold-out |

---

## 📦 Data

- **Source:** Yahoo Finance via `yfinance`
- **Asset:** AAPL
- **Period:** January 2020 – January 2025 (1,258 trading days)
- **Features:** Close, High, Low, Open, Volume → engineered into 15+ ML-ready features
- **Target:** Binary direction of next-day return (`shift(-1)`)

---

## 🗂️ Project Structure

```
stock-return-feature-engineering/
│
├── End-to-End_Feature_Engineering_Project_Stock_Return_Prediction.ipynb
└── README.md
```

---

## 🔧 Technical Stack

| Library | Purpose |
|---|---|
| `yfinance` | Live AAPL price data download |
| `pandas` | Feature construction, rolling windows, encoding |
| `numpy` | Log transforms, momentum calculations |
| `scipy.stats.boxcox` | Box-Cox power transformation |
| `sklearn.preprocessing` | StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder |
| `sklearn.pipeline` | Pipeline + ColumnTransformer assembly |
| `sklearn.impute` | SimpleImputer (mean strategy for numeric, most_frequent for categorical) |
| `sklearn.linear_model` | LogisticRegression baseline classifier |
| `sklearn.metrics` | accuracy_score, classification_report |
| `matplotlib` | Actual vs predicted visualisation |

---

## 📐 Feature Engineering Details

### Price-Based Features

```python
data['returns']       = close.pct_change()
data['log_returns']   = np.log(close).diff()
data['momentum_5']    = close - close.shift(5)
data['momentum_10']   = close - close.shift(10)
data['sma_10']        = close.rolling(10).mean()
data['sma_20']        = close.rolling(20).mean()
data['volatility_10'] = data['returns'].rolling(10).std()
data['volatility_20'] = data['returns'].rolling(20).std()
data['daily_range']   = high - low           # intraday price range
data['volume_change'] = volume.pct_change()  # volume momentum
```

**Financial rationale per feature:**
- `momentum_5/10` — captures short/medium-term trend direction; positive momentum signals continuation
- `sma_10/20` — trend identification; price relative to SMA is a mean-reversion signal
- `volatility_10/20` — current risk regime; used to scale position sizing in practice
- `daily_range` — intraday uncertainty; wide range days often precede directional moves
- `volume_change` — volume confirms or questions price moves; anomalous volume often precedes reversals

### Transformations

**Log transformation** on close price reduces right-skew from price levels growing exponentially over time.

**Box-Cox transformation** — automatically finds the optimal power λ to normalise the distribution:
```
λ = 1.007  ← very close to 1, meaning the close price is nearly log-linear
```
A λ near 1 confirms the log transform is appropriate and that no radical power correction is needed.

**Skewness audit across all features:**

| Feature | Skewness | Issue? |
|---|---|---|
| Volume | +2.348 | ⚠️ Right-skewed |
| volatility_10 | +2.666 | ⚠️ Right-skewed |
| returns | +0.106 | ✅ Near-zero |
| log_returns | −0.112 | ✅ Near-zero |
| Close (boxcox) | −0.028 | ✅ Near-zero after transform |

### Calendar Encoding

Three encoding strategies demonstrated on `day_of_week` and `month_name`:

**One-hot encoding** (`pd.get_dummies`, `drop_first=True`) — creates binary columns for each category, avoids ordinal assumption. Best for unordered categories like day names.

**Label encoding** — assigns integers (Friday=0, Monday=1, Thursday=2, Tuesday=3, Wednesday=4). Computationally efficient but implies false ordering.

**Target encoding** — maps each category to its mean target value:

| Day | Mean Target (% up days) |
|---|---|
| Friday | 56.7% |
| Monday | 55.4% |
| Tuesday | 52.7% |
| Wednesday | 51.6% |
| Thursday | 50.2% |

Friday and Monday show the highest historical up-day rates in the sample — a subtle calendar effect captured in a single float per row.

### Scaling Comparison

Three scalers applied to `['returns', 'momentum_5', 'momentum_10', 'volatility_10']`:

| Scaler | Formula | Best For |
|---|---|---|
| `StandardScaler` | `(x - μ) / σ` | Normally distributed features |
| `MinMaxScaler` | `(x - min) / (max - min)` | Bounded features, neural nets |
| `RobustScaler` | `(x - median) / IQR` | **Fat-tailed financial data** ← recommended |

`RobustScaler` is the correct default choice for financial returns because it uses the median and IQR rather than mean and std — making it insensitive to the extreme outliers (crash days, spike days) that are structurally present in financial time series.

### Pipeline Assembly

```python
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])
```

Output shape: **(1,258, 21)** — 4 numeric features + 17 OHE columns (5 days + 12 months, minus drop_first columns).

---

## 📊 Model Results

**Logistic Regression on 20% held-out test set (248 samples, no shuffle):**

```
Accuracy: 52.0%

              precision  recall  f1-score  support
          0       0.36    0.15      0.21      107
          1       0.55    0.80      0.66      141

   accuracy                         0.52      248
```

**Key observations:**
- The model predicts class 1 (up day) with 80% recall — it correctly captures most up days
- Class 0 (down day) recall of only 15% — the model rarely predicts down days correctly
- 52% accuracy is marginally above the 50% random baseline but not statistically significant

This is an expected and honest result. AAPL exhibits persistent upward bias over the 2020–2025 period (141 up days vs 107 down days in the test set), and the model has learned to lean long. Beating 52% on daily return direction consistently would represent genuine alpha.

---

## ⚠️ Production-Readiness Notes

**Train-test split without shuffle:**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
```
`shuffle=False` preserves temporal order — essential for time series to avoid lookahead bias. Shuffled splits would leak future data into training.

**Target encoding risk:** Target encoding on day-of-week without cross-validation introduces mild target leakage (the training target influences the encoding). In production this should be computed on training folds only.

**Scaler fit on training data only:**
```python
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)   # ← .transform only, not .fit_transform
```
Correct — the scaler parameters are learned from training data only and applied to test data without refitting.

---

## 🧠 Concepts Demonstrated

| Concept | Implementation |
|---|---|
| Financial feature construction | 10+ indicators from OHLCV — momentum, SMA, volatility, range, volume |
| Log & Box-Cox transformation | Distribution normalisation with λ estimation |
| Skewness audit | `data.skew()` across all features; identification of problem columns |
| Target variable construction | `shift(-1)` binary direction with no lookahead |
| Calendar encoding | OHE, label encoding, target encoding on day/month features |
| Scaler comparison | Standard, MinMax, Robust — with reasoning for RobustScaler in finance |
| sklearn Pipeline | `ColumnTransformer` + imputation + scaling + OHE in one object |
| Temporal train-test split | `shuffle=False` to prevent lookahead bias |
| Classification evaluation | Accuracy, precision, recall, F1 — with class-level interpretation |

---

## 🚀 How to Run

**Requirements:**
```
pandas numpy matplotlib yfinance scipy scikit-learn
```

```bash
pip install pandas numpy matplotlib yfinance scipy scikit-learn
```

Downloads live data from Yahoo Finance — no static files needed.

```bash
git clone https://github.com/Patience-Fuglo/stock-return-feature-engineering.git
cd stock-return-feature-engineering
jupyter notebook
```

---

## 📌 Context

Ninth project in a Python for Quantitative Finance series. Marks the shift from pure statistical analysis into **ML systems engineering** — the feature engineering and preprocessing layer that sits between raw market data and any predictive model. The honest 52% accuracy result and the detailed discussion of data leakage risks reflect the real challenges of applying ML to financial time series, where even marginal edge is difficult to achieve and easy to accidentally manufacture through lookahead bias.
