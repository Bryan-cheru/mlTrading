# Mathematical Functions for Institutional ML Trading

## 🧮 Overview

This document analyzes the mathematical and statistical functions that should replace technical indicators in our ML models, based on institutional quantitative finance practices.

## 📊 Categories of Mathematical Functions

### 1. **Statistical Measures**

#### **Z-Score (Standardization)**
```python
z_score = (x - mean(x)) / std(x)
```
- **Purpose**: Measure how many standard deviations a value is from the mean
- **Use Case**: Identify overbought/oversold conditions statistically
- **Advantage over RSI**: More mathematically rigorous, no arbitrary 70/30 thresholds

#### **Rolling Correlation**
```python
correlation = corr(series_A, series_B, window=n)
```
- **Purpose**: Measure linear relationship between two instruments
- **Use Case**: Identify pairs trading opportunities, regime changes
- **Advantage over MACD**: Direct statistical relationship measurement

#### **Cointegration Test**
```python
# Engle-Granger test
residuals = prices_A - beta * prices_B
adf_statistic = augmented_dickey_fuller(residuals)
```
- **Purpose**: Test if two non-stationary series have a stable long-term relationship
- **Use Case**: Validate pairs trading relationships
- **Advantage**: Statistical significance testing

### 2. **Probability and Distribution Functions**

#### **Probability Density Functions**
```python
# Normal distribution probability
prob_normal = norm.pdf(returns, loc=mean, scale=std)

# Student's t-distribution (fat tails)
prob_t = t.pdf(returns, df=degrees_freedom)
```
- **Purpose**: Model return distributions, estimate probabilities
- **Use Case**: Risk assessment, outlier detection
- **Advantage**: Captures actual return distributions vs simple price ratios

#### **Cumulative Distribution Functions**
```python
# Probability of observing return <= x
prob_cumulative = norm.cdf(x, loc=mean, scale=std)
```
- **Purpose**: Calculate percentiles and probability thresholds
- **Use Case**: Position sizing based on confidence intervals

#### **Quantile Functions**
```python
# Value at Risk (VaR) calculation
var_95 = np.percentile(returns, 5)  # 5th percentile
var_99 = np.percentile(returns, 1)  # 1st percentile
```
- **Purpose**: Risk measurement and portfolio allocation
- **Use Case**: Determine position sizes based on statistical risk

### 3. **Time Series Mathematical Functions**

#### **Autoregressive (AR) Components**
```python
# AR(1) model: X_t = φ * X_{t-1} + ε_t
ar_coefficient = np.corrcoef(series[1:], series[:-1])[0,1]
ar_residuals = series[1:] - ar_coefficient * series[:-1]
```
- **Purpose**: Model temporal dependencies
- **Use Case**: Predict mean reversion patterns

#### **GARCH Volatility Models**
```python
# Generalized Autoregressive Conditional Heteroskedasticity
σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}
```
- **Purpose**: Model time-varying volatility
- **Use Case**: Dynamic position sizing, risk adjustment
- **Advantage over ATR**: Statistically sound volatility forecasting

#### **Kalman Filter State Estimation**
```python
# State space model for dynamic regression
β_t = β_{t-1} + w_t  # Time-varying coefficient
y_t = x_t * β_t + v_t  # Observation equation
```
- **Purpose**: Estimate time-varying relationships
- **Use Case**: Dynamic hedge ratios, adaptive strategies

### 4. **Information Theory Functions**

#### **Entropy Measures**
```python
# Shannon entropy
entropy = -sum(p * log(p)) for p in probabilities
```
- **Purpose**: Measure information content and uncertainty
- **Use Case**: Market regime identification, signal strength

#### **Mutual Information**
```python
# Information shared between two variables
mutual_info = entropy(X) + entropy(Y) - entropy(X,Y)
```
- **Purpose**: Measure non-linear dependencies
- **Use Case**: Feature selection, correlation beyond linear relationships

### 5. **Fourier and Signal Processing**

#### **Discrete Fourier Transform (DFT)**
```python
# Frequency domain analysis
fft_coefficients = np.fft.fft(price_series)
dominant_frequencies = np.fft.fftfreq(len(price_series))
```
- **Purpose**: Identify cyclical patterns and dominant frequencies
- **Use Case**: Market cycle analysis, seasonal patterns

#### **Wavelet Transform**
```python
# Multi-resolution analysis
coefficients = pywt.wavedec(prices, 'db4', level=5)
```
- **Purpose**: Time-frequency analysis
- **Use Case**: Identify patterns at different time scales

### 6. **Optimization and Mathematical Programming**

#### **Linear Programming Solutions**
```python
# Portfolio optimization
result = linprog(c, A_ub, b_ub, bounds=bounds)
optimal_weights = result.x
```
- **Purpose**: Optimal allocation under constraints
- **Use Case**: Position sizing, risk budgeting

#### **Quadratic Programming (Markowitz)**
```python
# Mean-variance optimization
def objective(weights):
    return weights.T @ cov_matrix @ weights
```
- **Purpose**: Risk-return optimization
- **Use Case**: Portfolio construction

### 7. **Stochastic Processes**

#### **Ornstein-Uhlenbeck Process**
```python
# Mean-reverting process
dX_t = θ(μ - X_t)dt + σ dW_t
```
- **Purpose**: Model mean-reverting behavior
- **Use Case**: Pairs trading, statistical arbitrage

#### **Jump Diffusion Models**
```python
# Merton jump-diffusion
dS = μS dt + σS dW + S(e^J - 1)dN
```
- **Purpose**: Model sudden price movements
- **Use Case**: Risk management, option pricing

## 🎯 **Practical Implementation Strategy**

### **Replace Technical Indicators with Mathematical Functions**

| **Technical Indicator** | **Mathematical Replacement** | **Advantage** |
|------------------------|------------------------------|---------------|
| RSI | Z-score of returns | Statistical significance |
| MACD | Cross-correlation function | Lag relationship quantification |
| Bollinger Bands | Confidence intervals (95%, 99%) | Probability-based thresholds |
| Stochastic | Percentile rank function | Distribution-based positioning |
| Williams %R | Quantile function | Statistical extremes |
| ATR | GARCH volatility forecast | Dynamic volatility modeling |

### **Feature Engineering with Mathematical Functions**

```python
def create_mathematical_features(prices, returns):
    features = {}
    
    # 1. Statistical measures
    features['z_score_5'] = (prices - prices.rolling(5).mean()) / prices.rolling(5).std()
    features['z_score_20'] = (prices - prices.rolling(20).mean()) / prices.rolling(20).std()
    
    # 2. Distribution features
    features['skewness'] = returns.rolling(20).skew()
    features['kurtosis'] = returns.rolling(20).kurt()
    features['var_95'] = returns.rolling(20).quantile(0.05)
    
    # 3. Time series features
    features['autocorr_1'] = returns.rolling(20).apply(lambda x: x.autocorr(lag=1))
    features['hurst_exponent'] = calculate_hurst_exponent(prices)
    
    # 4. Information theory
    features['entropy'] = returns.rolling(20).apply(calculate_entropy)
    
    # 5. Volatility modeling
    features['garch_vol'] = calculate_garch_volatility(returns)
    
    return features
```

## 🔬 **Advanced Mathematical Concepts**

### **1. Fractional Calculus**
- **Fractional derivatives**: Better modeling of memory effects in financial time series
- **Long-range dependence**: Capture persistent patterns traditional calculus misses

### **2. Machine Learning Mathematical Functions**
- **Kernel functions**: RBF, polynomial kernels for non-linear pattern recognition
- **Distance metrics**: Mahalanobis distance for multivariate outlier detection
- **Similarity measures**: Cosine similarity, dynamic time warping

### **3. Network Theory**
- **Graph centrality measures**: Identify important market relationships
- **Clustering coefficients**: Market structure analysis
- **Network entropy**: Systemic risk measurement

## 📊 **Implementation Priority**

### **High Priority (Immediate Implementation)**
1. **Z-scores** (replace RSI, Stochastic)
2. **Rolling correlations** (replace MACD)
3. **Quantile functions** (replace Bollinger Bands)
4. **GARCH volatility** (replace ATR)
5. **Autocorrelation** (momentum measurement)

### **Medium Priority (Phase 2)**
1. **Cointegration tests** (pairs validation)
2. **Information theory measures** (signal strength)
3. **Fourier analysis** (cycle identification)
4. **Kalman filtering** (adaptive parameters)

### **Advanced (Phase 3)**
1. **Stochastic process modeling**
2. **Jump diffusion detection**
3. **Fractional calculus features**
4. **Network analysis**

## 🎯 **Expected Improvements**

### **Advantages of Mathematical Functions**
- **Statistical Rigor**: P-values, confidence intervals, hypothesis testing
- **Adaptive Parameters**: Time-varying coefficients vs fixed indicator periods
- **Theoretical Foundation**: Based on financial mathematics vs empirical rules
- **Risk Quantification**: Probability-based risk measures
- **Model Validation**: Statistical tests for model adequacy

### **Performance Enhancement**
- **Better Feature Quality**: Statistically meaningful vs ad-hoc indicators
- **Reduced Overfitting**: Mathematical constraints vs arbitrary parameters
- **Regime Adaptability**: Statistical models adapt to changing market conditions
- **Risk-Adjusted Returns**: Mathematical optimization of risk-return trade-offs

This mathematical foundation will provide the institutional-grade feature engineering needed for successful ML-based trading while maintaining the client's preference for statistical rigor over traditional technical analysis.