"""
ES ML Trading System - Model Training Pipeline
Institutional-grade ML model training for ES futures trading
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

class ESMLModelTrainer:
    """
    Professional ML model trainer for ES futures trading
    Uses institutional-grade features and validation methods
    """
    
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()  # Better for financial data than StandardScaler
        self.feature_columns = []
        self.is_trained = False
        
    def download_training_data(self, symbol="ES=F", period="2y"):
        """
        Download ES futures historical data for training
        
        Args:
            symbol: ES futures symbol (ES=F for continuous contract)
            period: Data period (1y, 2y, 5y, max)
        """
        print(f"📊 Downloading {period} of ES futures data...")
        
        try:
            # Download ES futures data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval="1h")  # 1-hour bars
            
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
                
            print(f"✅ Downloaded {len(data)} bars of data")
            print(f"📅 Date range: {data.index[0]} to {data.index[-1]}")
            
            return data
            
        except Exception as e:
            print(f"❌ Error downloading data: {e}")
            return None
    
    def engineer_features(self, data):
        """
        Create comprehensive technical features for ML training
        Institutional-grade feature engineering
        """
        print("🔧 Engineering features...")
        
        df = data.copy()
        
        # Price-based features
        df['Returns_1'] = df['Close'].pct_change(1)
        df['Returns_5'] = df['Close'].pct_change(5)
        df['Returns_10'] = df['Close'].pct_change(10)
        
        # Moving averages and crossovers
        df['SMA_10'] = df['Close'].rolling(10).mean()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_Ratio_10_20'] = df['SMA_10'] / df['SMA_20']
        df['SMA_Ratio_20_50'] = df['SMA_20'] / df['SMA_50']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (2 * bb_std)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volatility features
        df['Volatility_10'] = df['Returns_1'].rolling(10).std()
        df['Volatility_20'] = df['Returns_1'].rolling(20).std()
        
        # Volume features (if available)
        if 'Volume' in df.columns:
            df['Volume_MA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Price momentum
        df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Support/Resistance levels
        df['High_20'] = df['High'].rolling(20).max()
        df['Low_20'] = df['Low'].rolling(20).min()
        df['Price_Position'] = (df['Close'] - df['Low_20']) / (df['High_20'] - df['Low_20'])
        
        # Time-based features
        df['Hour'] = df.index.hour
        df['DayOfWeek'] = df.index.dayofweek
        df['IsMarketOpen'] = ((df['Hour'] >= 9) & (df['Hour'] <= 16)).astype(int)
        
        print(f"✅ Created {len([col for col in df.columns if col not in data.columns])} new features")
        
        return df
    
    def create_labels(self, data, forward_periods=5, threshold=0.001):
        """
        Create trading labels based on future price movements
        
        Args:
            data: DataFrame with OHLCV data
            forward_periods: Look-ahead periods for labeling
            threshold: Minimum return threshold for BUY/SELL signals
        """
        print(f"🏷️ Creating labels (look-ahead: {forward_periods} periods, threshold: {threshold:.3f})...")
        
        df = data.copy()
        
        # Calculate future returns
        future_returns = df['Close'].shift(-forward_periods) / df['Close'] - 1
        
        # Create labels
        conditions = [
            future_returns > threshold,   # BUY signal
            future_returns < -threshold,  # SELL signal
        ]
        choices = [1, -1]  # 1 = BUY, -1 = SELL, 0 = HOLD
        
        df['Label'] = np.select(conditions, choices, default=0)
        
        # Label distribution
        label_counts = df['Label'].value_counts().sort_index()
        print("📊 Label distribution:")
        labels = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {labels.get(label, label)}: {count:,} ({percentage:.1f}%)")
        
        return df
    
    def prepare_training_data(self, data):
        """
        Prepare final training dataset with feature selection
        """
        print("🔄 Preparing training data...")
        
        # Select features (exclude OHLCV and label columns)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Label', 'Adj Close']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # Remove any remaining NaN values
        df_clean = data[feature_cols + ['Label']].dropna()
        
        X = df_clean[feature_cols]
        y = df_clean['Label']
        
        self.feature_columns = feature_cols
        
        print(f"✅ Training data prepared:")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Samples: {len(X):,}")
        print(f"   Date range: {df_clean.index[0]} to {df_clean.index[-1]}")
        
        return X, y
    
    def train_model(self, X, y, n_splits=5):
        """
        Train Random Forest model with time series cross-validation
        """
        print("🤖 Training ML model...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize model with institutional-grade parameters
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Handle class imbalance
        )
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=tscv, scoring='accuracy')
        
        print(f"📈 Cross-validation scores: {cv_scores}")
        print(f"📊 Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Train final model on all data
        self.model.fit(X_scaled, y)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("🔍 Top 10 most important features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
        
        self.is_trained = True
        print("✅ Model training completed!")
        
        return cv_scores
    
    def evaluate_model(self, X, y):
        """
        Detailed model evaluation
        """
        if not self.is_trained:
            print("❌ Model not trained yet!")
            return
        
        print("📋 Model evaluation...")
        
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        
        # Classification report
        print("\n📊 Classification Report:")
        labels = [-1, 0, 1]
        target_names = ['SELL', 'HOLD', 'BUY']
        print(classification_report(y, y_pred, labels=labels, target_names=target_names))
        
        # Confusion matrix
        print("\n🔢 Confusion Matrix:")
        cm = confusion_matrix(y, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
        print(cm_df)
        
        # Trading performance simulation
        self.simulate_trading_performance(X, y)
    
    def simulate_trading_performance(self, X, y):
        """
        Simulate trading performance with the trained model
        """
        print("\n💰 Trading Performance Simulation...")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Calculate confidence (max probability)
        confidence = np.max(probabilities, axis=1)
        
        # Filter high-confidence trades (>70%)
        high_conf_mask = confidence > 0.7
        high_conf_predictions = predictions[high_conf_mask]
        high_conf_actual = y.iloc[np.where(high_conf_mask)[0]]
        
        if len(high_conf_predictions) > 0:
            accuracy = (high_conf_predictions == high_conf_actual).mean()
            print(f"📈 High-confidence trades (>70%): {len(high_conf_predictions)}")
            print(f"🎯 High-confidence accuracy: {accuracy:.3f}")
            
            # Signal distribution for high-confidence trades
            unique, counts = np.unique(high_conf_predictions, return_counts=True)
            signal_dist = dict(zip(unique, counts))
            labels = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
            print("📊 High-confidence signal distribution:")
            for signal, count in signal_dist.items():
                print(f"   {labels.get(signal, signal)}: {count}")
        else:
            print("⚠️ No high-confidence trades found")
    
    def save_model(self, filepath="models/es_ml_model.joblib"):
        """
        Save trained model and scaler
        """
        if not self.is_trained:
            print("❌ No trained model to save!")
            return
        
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'trained_date': pd.Timestamp.now()
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to: {filepath}")
    
    def load_model(self, filepath="models/es_ml_model.joblib"):
        """
        Load trained model and scaler
        """
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = True
            
            trained_date = model_data.get('trained_date', 'Unknown')
            print(f"📥 Model loaded from: {filepath}")
            print(f"🗓️ Trained on: {trained_date}")
            print(f"🔧 Features: {len(self.feature_columns)}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def predict(self, features):
        """
        Make predictions with confidence scores
        """
        if not self.is_trained:
            print("❌ Model not trained!")
            return None, 0.0
        
        # Ensure features are in correct order
        X = features[self.feature_columns]
        X_scaled = self.scaler.transform([X])
        
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        confidence = np.max(probabilities)
        
        # Convert to signal names
        signal_map = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
        signal = signal_map[prediction]
        
        return signal, confidence


def train_es_model():
    """
    Complete ES ML model training pipeline
    """
    print("🚀 Starting ES ML Model Training Pipeline...")
    print("=" * 60)
    
    # Initialize trainer
    trainer = ESMLModelTrainer()
    
    # Download data
    data = trainer.download_training_data(symbol="ES=F", period="2y")
    if data is None:
        return None
    
    # Engineer features
    featured_data = trainer.engineer_features(data)
    
    # Create labels
    labeled_data = trainer.create_labels(featured_data, forward_periods=5, threshold=0.001)
    
    # Prepare training data
    X, y = trainer.prepare_training_data(labeled_data)
    
    # Train model
    cv_scores = trainer.train_model(X, y)
    
    # Evaluate model
    trainer.evaluate_model(X, y)
    
    # Save model
    trainer.save_model()
    
    print("\n🎉 Training pipeline completed!")
    print("=" * 60)
    
    return trainer


if __name__ == "__main__":
    # Run the training pipeline
    trainer = train_es_model()