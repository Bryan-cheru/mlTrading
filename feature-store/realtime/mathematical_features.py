"""
Mathematical Feature Engineering for Institutional ML Trading
Replaces technical indicators with mathematical/statistical functions
Based on quantitative finance and statistical analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import scipy.stats as stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MarketData:
    """Market data container"""
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: int

@dataclass
class FeatureSet:
    """Feature set container with mathematical features"""
    timestamp: pd.Timestamp
    features: Dict[str, float]
    target: Optional[float] = None

class MathematicalFeatureEngine:
    """
    Mathematical feature engineering for institutional ML trading
    Replaces technical indicators with statistical/mathematical functions
    Based on quantitative finance principles
    """
    
    def __init__(self, lookback_periods: int = 300):
        self.lookback_periods = lookback_periods
        self.price_buffer = deque(maxlen=lookback_periods)
        self.volume_buffer = deque(maxlen=lookback_periods)
        self.returns_buffer = deque(maxlen=lookback_periods)
        self.feature_cache = {}
        
        # Mathematical computation caches
        self.correlation_cache = {}
        self.volatility_cache = {}
        self.distribution_cache = {}
        
        # Pre-compute mathematical constants
        self.sqrt_252 = np.sqrt(252)  # Annualization factor
        self.log_2 = np.log(2)        # For entropy calculations
        
        print(f"🧮 Mathematical Feature Engine initialized with {lookback_periods} period lookback")
    
    def update_buffers(self, market_data: MarketData):
        """Update rolling buffers with new market data"""
        price_data = {
            'timestamp': market_data.timestamp,
            'open': market_data.open,
            'high': market_data.high,
            'low': market_data.low,
            'close': market_data.close
        }
        
        self.price_buffer.append(price_data)
        self.volume_buffer.append(market_data.volume)
        
        # Calculate and store returns
        if len(self.price_buffer) >= 2:
            prev_close = self.price_buffer[-2]['close']
            current_close = market_data.close
            if prev_close > 0:
                log_return = np.log(current_close / prev_close)
                self.returns_buffer.append(log_return)
    
    def compute_statistical_measures(self) -> Dict[str, float]:
        """
        Compute statistical measures to replace technical indicators
        Z-scores, correlations, distribution parameters
        """
        if len(self.price_buffer) < 50:
            return {}
        
        features = {}
        prices = np.array([bar['close'] for bar in self.price_buffer])
        returns = np.array(list(self.returns_buffer))
        
        # === Z-SCORES (Replace RSI, Stochastic) ===
        for window in [20, 50, 100]:
            if len(prices) >= window:
                recent_prices = prices[-window:]
                mean_price = np.mean(recent_prices)
                std_price = np.std(recent_prices)
                
                if std_price > 0:
                    z_score = (prices[-1] - mean_price) / std_price
                    features[f'z_score_price_{window}'] = z_score
                    
                # Z-score of returns
                if len(returns) >= window:
                    recent_returns = returns[-window:]
                    mean_return = np.mean(recent_returns)
                    std_return = np.std(recent_returns)
                    
                    if std_return > 0 and len(returns) > 0:
                        z_score_return = (returns[-1] - mean_return) / std_return
                        features[f'z_score_return_{window}'] = z_score_return
        
        # === ROLLING CORRELATION (Replace MACD) ===
        if len(prices) >= 40:
            # Price vs lagged price correlation (momentum proxy)
            for lag in [1, 5, 10]:
                if len(prices) > lag:
                    correlation = np.corrcoef(prices[lag:], prices[:-lag])[0, 1]
                    if not np.isnan(correlation):
                        features[f'autocorr_lag_{lag}'] = correlation
            
            # Rolling correlation with moving average
            for window in [20, 50]:
                if len(prices) >= window:
                    ma = np.mean(prices[-window:])
                    price_series = prices[-window:]
                    ma_series = np.full(window, ma)
                    correlation = np.corrcoef(price_series, ma_series)[0, 1]
                    if not np.isnan(correlation):
                        features[f'price_ma_corr_{window}'] = correlation
        
        # === DISTRIBUTION STATISTICS ===
        if len(returns) >= 50:
            recent_returns = returns[-50:]
            
            # Moments of return distribution
            features['returns_mean'] = np.mean(recent_returns)
            features['returns_std'] = np.std(recent_returns)
            features['returns_skewness'] = stats.skew(recent_returns)
            features['returns_kurtosis'] = stats.kurtosis(recent_returns)
            
            # Normality test (Jarque-Bera)
            try:
                jb_stat, jb_pvalue = stats.jarque_bera(recent_returns)
                features['jarque_bera_stat'] = jb_stat
                features['jarque_bera_pvalue'] = jb_pvalue
            except:
                features['jarque_bera_stat'] = 0
                features['jarque_bera_pvalue'] = 1
        
        return features
    
    def compute_probability_functions(self) -> Dict[str, float]:
        """
        Compute probability-based features
        VaR, confidence intervals, probability distributions
        """
        if len(self.returns_buffer) < 30:
            return {}
        
        features = {}
        returns = np.array(list(self.returns_buffer))
        
        # === VALUE AT RISK (VaR) ===
        for confidence in [95, 99]:
            percentile = 100 - confidence
            var = np.percentile(returns, percentile)
            features[f'var_{confidence}'] = var
            
            # Expected Shortfall (Conditional VaR)
            es = np.mean(returns[returns <= var]) if np.any(returns <= var) else var
            features[f'expected_shortfall_{confidence}'] = es
        
        # === QUANTILE FUNCTIONS (Replace Bollinger Bands) ===
        for quantile in [0.05, 0.10, 0.25, 0.75, 0.90, 0.95]:
            q_value = np.quantile(returns, quantile)
            features[f'quantile_{int(quantile*100)}'] = q_value
        
        # Current return percentile rank
        if len(returns) > 0:
            current_return = returns[-1]
            percentile_rank = stats.percentileofscore(returns, current_return)
            features['return_percentile_rank'] = percentile_rank / 100.0
        
        # === PROBABILITY DENSITY ESTIMATION ===
        if len(returns) >= 50:
            try:
                # Fit normal distribution
                mu_norm, sigma_norm = stats.norm.fit(returns)
                features['normal_mu'] = mu_norm
                features['normal_sigma'] = sigma_norm
                
                # Current return probability under normal distribution
                if len(returns) > 0:
                    prob_normal = stats.norm.pdf(returns[-1], mu_norm, sigma_norm)
                    features['prob_density_normal'] = prob_normal
                
                # Fit Student's t-distribution (better for financial returns)
                df_t, mu_t, sigma_t = stats.t.fit(returns)
                features['t_dist_df'] = df_t
                features['t_dist_mu'] = mu_t
                features['t_dist_sigma'] = sigma_t
                
                # Current return probability under t-distribution
                if len(returns) > 0:
                    prob_t = stats.t.pdf(returns[-1], df_t, mu_t, sigma_t)
                    features['prob_density_t'] = prob_t
                    
            except Exception as e:
                print(f"Warning: Distribution fitting failed: {e}")
        
        return features
    
    def compute_time_series_functions(self) -> Dict[str, float]:
        """
        Compute time series mathematical functions
        Autoregressive, GARCH, Kalman filter components
        """
        if len(self.returns_buffer) < 50:
            return {}
        
        features = {}
        returns = np.array(list(self.returns_buffer))
        prices = np.array([bar['close'] for bar in self.price_buffer])
        
        # === AUTOREGRESSIVE FEATURES ===
        for lag in [1, 2, 5]:
            if len(returns) > lag:
                # AR coefficient estimation
                y = returns[lag:]
                x = returns[:-lag]
                if len(y) > 0 and np.std(x) > 0:
                    correlation = np.corrcoef(y, x)[0, 1]
                    features[f'ar_coeff_lag_{lag}'] = correlation
                    
                    # AR residuals
                    predicted = correlation * x
                    residuals = y - predicted
                    features[f'ar_residual_std_lag_{lag}'] = np.std(residuals)
        
        # === GARCH-STYLE VOLATILITY (Replace ATR) ===
        if len(returns) >= 20:
            # Squared returns (volatility proxy)
            squared_returns = returns ** 2
            
            # Exponentially weighted moving variance
            for alpha in [0.06, 0.1, 0.2]:  # GARCH-like decay factors
                ewm_var = self._exponential_weighted_variance(squared_returns, alpha)
                features[f'ewm_volatility_alpha_{int(alpha*100)}'] = np.sqrt(ewm_var * 252)
            
            # Volatility clustering measure
            vol_series = np.sqrt(squared_returns)
            if len(vol_series) >= 10:
                vol_autocorr = np.corrcoef(vol_series[1:], vol_series[:-1])[0, 1]
                if not np.isnan(vol_autocorr):
                    features['volatility_clustering'] = vol_autocorr
        
        # === MEAN REVERSION TESTS ===
        if len(prices) >= 50:
            # Augmented Dickey-Fuller test approximation
            # (Full ADF test too slow for real-time, using simplified version)
            log_prices = np.log(prices)
            price_diff = np.diff(log_prices)
            lagged_prices = log_prices[:-1]
            
            if len(price_diff) > 0 and np.std(lagged_prices) > 0:
                # Simple mean reversion coefficient
                mean_reversion_coeff = np.corrcoef(price_diff, lagged_prices)[0, 1]
                if not np.isnan(mean_reversion_coeff):
                    features['mean_reversion_coeff'] = mean_reversion_coeff
        
        # === HALF-LIFE OF MEAN REVERSION ===
        if len(prices) >= 100:
            try:
                log_prices = np.log(prices[-100:])
                # Ornstein-Uhlenbeck process parameter estimation
                half_life = self._estimate_half_life(log_prices)
                if half_life > 0:
                    features['mean_reversion_half_life'] = half_life
            except:
                pass
        
        return features
    
    def compute_information_theory_functions(self) -> Dict[str, float]:
        """
        Compute information theory features
        Entropy, mutual information, complexity measures
        """
        if len(self.returns_buffer) < 50:
            return {}
        
        features = {}
        returns = np.array(list(self.returns_buffer))
        
        # === SHANNON ENTROPY ===
        try:
            # Discretize returns for entropy calculation
            n_bins = min(20, len(returns) // 3)
            hist, _ = np.histogram(returns, bins=n_bins, density=True)
            hist = hist[hist > 0]  # Remove zero probabilities
            
            if len(hist) > 1:
                # Normalize to get probabilities
                probabilities = hist / np.sum(hist)
                entropy = -np.sum(probabilities * np.log2(probabilities))
                features['shannon_entropy'] = entropy
                
                # Relative entropy (vs uniform distribution)
                uniform_entropy = np.log2(len(probabilities))
                features['relative_entropy'] = entropy / uniform_entropy if uniform_entropy > 0 else 0
        except:
            pass
        
        # === COMPLEXITY MEASURES ===
        if len(returns) >= 30:
            # Approximate entropy (regularity measure)
            try:
                approx_entropy = self._approximate_entropy(returns, m=2, r=0.2*np.std(returns))
                features['approximate_entropy'] = approx_entropy
            except:
                pass
            
            # Lempel-Ziv complexity (simplified)
            binary_returns = (returns > np.median(returns)).astype(int)
            lz_complexity = self._lempel_ziv_complexity(binary_returns)
            features['lz_complexity'] = lz_complexity
        
        return features
    
    def compute_fourier_analysis(self) -> Dict[str, float]:
        """
        Compute Fourier transform features
        Frequency domain analysis, cyclical patterns
        """
        if len(self.price_buffer) < 64:  # Need power of 2 for efficient FFT
            return {}
        
        features = {}
        prices = np.array([bar['close'] for bar in self.price_buffer])
        
        # === DISCRETE FOURIER TRANSFORM ===
        try:
            # Use recent data for FFT
            recent_prices = prices[-64:] if len(prices) >= 64 else prices
            log_prices = np.log(recent_prices)
            
            # Remove trend for better frequency analysis
            detrended = log_prices - np.linspace(log_prices[0], log_prices[-1], len(log_prices))
            
            # Compute FFT
            fft = np.fft.fft(detrended)
            freqs = np.fft.fftfreq(len(detrended))
            
            # Power spectral density
            psd = np.abs(fft) ** 2
            
            # Dominant frequency components
            positive_freqs = freqs[:len(freqs)//2]
            positive_psd = psd[:len(psd)//2]
            
            if len(positive_psd) > 0:
                # Most dominant frequency
                dominant_freq_idx = np.argmax(positive_psd[1:]) + 1  # Skip DC component
                features['dominant_frequency'] = positive_freqs[dominant_freq_idx]
                features['dominant_frequency_power'] = positive_psd[dominant_freq_idx]
                
                # Spectral centroid (frequency "center of mass")
                if np.sum(positive_psd) > 0:
                    spectral_centroid = np.sum(positive_freqs * positive_psd) / np.sum(positive_psd)
                    features['spectral_centroid'] = spectral_centroid
                
                # Spectral entropy
                psd_norm = positive_psd / np.sum(positive_psd)
                psd_norm = psd_norm[psd_norm > 0]
                if len(psd_norm) > 1:
                    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm))
                    features['spectral_entropy'] = spectral_entropy
                    
        except Exception as e:
            print(f"Warning: FFT analysis failed: {e}")
        
        return features
    
    def compute_optimization_features(self) -> Dict[str, float]:
        """
        Compute optimization-based features
        Risk metrics, portfolio-style calculations
        """
        if len(self.returns_buffer) < 30:
            return {}
        
        features = {}
        returns = np.array(list(self.returns_buffer))
        
        # === RISK-ADJUSTED METRICS ===
        if len(returns) >= 20:
            # Sharpe ratio (rolling)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return > 0:
                sharpe_ratio = mean_return / std_return * self.sqrt_252
                features['sharpe_ratio'] = sharpe_ratio
            
            # Sortino ratio (downside risk)
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0:
                downside_std = np.std(negative_returns)
                if downside_std > 0:
                    sortino_ratio = mean_return / downside_std * self.sqrt_252
                    features['sortino_ratio'] = sortino_ratio
            
            # Calmar ratio (return/max drawdown)
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            if max_drawdown < 0:
                calmar_ratio = (mean_return * 252) / abs(max_drawdown)
                features['calmar_ratio'] = calmar_ratio
                features['max_drawdown'] = max_drawdown
        
        # === KELLY CRITERION ===
        if len(returns) >= 20:
            # Simplified Kelly fraction
            win_rate = np.mean(returns > 0)
            if win_rate > 0 and win_rate < 1:
                avg_win = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0
                avg_loss = abs(np.mean(returns[returns < 0])) if np.any(returns < 0) else 1e-10
                
                if avg_loss > 0:
                    kelly_fraction = win_rate - (1 - win_rate) / (avg_win / avg_loss)
                    features['kelly_fraction'] = np.clip(kelly_fraction, -1, 1)
        
        return features
    
    def generate_features(self, market_data: MarketData) -> FeatureSet:
        """
        Generate complete mathematical feature set
        Main entry point for feature generation
        """
        self.update_buffers(market_data)
        
        if len(self.price_buffer) < 20:
            return FeatureSet(
                timestamp=market_data.timestamp,
                features={}
            )
        
        # Combine all mathematical feature groups
        all_features = {}
        
        # Core statistical measures
        all_features.update(self.compute_statistical_measures())
        
        # Probability functions
        all_features.update(self.compute_probability_functions())
        
        # Time series functions
        all_features.update(self.compute_time_series_functions())
        
        # Information theory
        all_features.update(self.compute_information_theory_functions())
        
        # Fourier analysis
        all_features.update(self.compute_fourier_analysis())
        
        # Optimization features
        all_features.update(self.compute_optimization_features())
        
        # Basic price features
        all_features.update(self._compute_basic_features(market_data))
        
        # Remove any NaN or infinite values
        cleaned_features = {}
        for key, value in all_features.items():
            if np.isfinite(value):
                cleaned_features[key] = float(value)
        
        return FeatureSet(
            timestamp=market_data.timestamp,
            features=cleaned_features
        )
    
    def _compute_basic_features(self, market_data: MarketData) -> Dict[str, float]:
        """Compute basic price and volume features"""
        features = {}
        
        if len(self.price_buffer) >= 2:
            prev_close = self.price_buffer[-2]['close']
            current_close = market_data.close
            
            # Basic features
            features['price'] = current_close
            features['log_return'] = np.log(current_close / prev_close) if prev_close > 0 else 0
            features['volume'] = market_data.volume
            
            # Price ratios
            features['high_low_ratio'] = market_data.high / market_data.low if market_data.low > 0 else 1
            features['close_open_ratio'] = current_close / market_data.open if market_data.open > 0 else 1
        
        return features
    
    # === HELPER FUNCTIONS ===
    
    def _exponential_weighted_variance(self, squared_returns: np.ndarray, alpha: float) -> float:
        """Calculate exponentially weighted variance (GARCH-style)"""
        if len(squared_returns) == 0:
            return 0.0
        
        weights = np.array([(1-alpha)**i for i in range(len(squared_returns))])
        weights = weights[::-1]  # Reverse to give more weight to recent observations
        weights = weights / np.sum(weights)  # Normalize
        
        return np.sum(weights * squared_returns)
    
    def _estimate_half_life(self, log_prices: np.ndarray) -> float:
        """Estimate half-life of mean reversion"""
        try:
            # Simple Ornstein-Uhlenbeck estimation
            price_diff = np.diff(log_prices)
            lagged_prices = log_prices[:-1] - np.mean(log_prices[:-1])
            
            if len(price_diff) > 0 and np.std(lagged_prices) > 0:
                # OLS regression: Δp_t = α + β * p_{t-1} + ε_t
                beta = np.cov(price_diff, lagged_prices)[0, 1] / np.var(lagged_prices)
                
                if beta < 0:  # Mean reverting
                    half_life = -np.log(2) / beta
                    return min(half_life, 1000)  # Cap at reasonable value
            
            return 0.0
        except:
            return 0.0
    
    def _approximate_entropy(self, data: np.ndarray, m: int, r: float) -> float:
        """Calculate approximate entropy"""
        try:
            N = len(data)
            
            def _maxdist(xi, xj, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _phi(m):
                patterns = np.array([data[i:i+m] for i in range(N - m + 1)])
                C = np.zeros(N - m + 1)
                
                for i in range(N - m + 1):
                    template = patterns[i]
                    matches = sum([1 for j in range(N - m + 1) 
                                 if _maxdist(template, patterns[j], m) <= r])
                    C[i] = matches / (N - m + 1.0)
                
                phi = np.mean([np.log(c) for c in C if c > 0])
                return phi
            
            return _phi(m) - _phi(m + 1)
        except:
            return 0.0
    
    def _lempel_ziv_complexity(self, binary_sequence: np.ndarray) -> float:
        """Calculate Lempel-Ziv complexity"""
        try:
            s = ''.join(binary_sequence.astype(str))
            n = len(s)
            complexity = 0
            i = 0
            
            while i < n:
                substring = s[i]
                j = 1
                
                while i + j <= n:
                    if substring in s[:i]:
                        substring = s[i:i+j]
                        j += 1
                    else:
                        break
                
                complexity += 1
                i += max(1, j-1)
            
            return complexity / n if n > 0 else 0
        except:
            return 0.0

# Create global instance for backward compatibility
mathematical_feature_engine = MathematicalFeatureEngine()

def generate_mathematical_features(market_data: MarketData) -> FeatureSet:
    """Generate mathematical features - main entry point"""
    return mathematical_feature_engine.generate_features(market_data)