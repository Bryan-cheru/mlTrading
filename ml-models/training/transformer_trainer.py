"""
ES Transformer Model Trainer
Production-ready transformer for ES futures trading
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import yfinance as yf
import joblib
import json
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import math
from pathlib import Path

# Import the base transformer model using direct path
import importlib.util
transformer_path = os.path.join(os.path.dirname(__file__), '..', 'advanced', 'transformer_model.py')
spec = importlib.util.spec_from_file_location("transformer_model", transformer_path)
transformer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transformer_module)
TradingTransformer = transformer_module.TradingTransformer
PositionalEncoding = transformer_module.PositionalEncoding

class ESTransformerDataset(Dataset):
    """Dataset for ES trading transformer"""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.LongTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class ESTransformerTrainer:
    """
    Complete ES Transformer Training Pipeline
    """
    
    def __init__(self, device=None):
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"🔧 Using device: {self.device}")
        
        # Model parameters
        self.seq_length = 100  # 100 bars = ~1.5 hours
        self.input_dim = 25    # Number of features
        self.d_model = 256     # Transformer dimension
        self.nhead = 8         # Attention heads
        self.num_layers = 6    # Transformer layers
        
        # Training parameters
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.epochs = 50
        
        # Model components
        self.model = None
        self.scaler = RobustScaler()
        self.feature_columns = []
        
        # Paths
        self.model_dir = Path(__file__).parent.parent.parent / "models"
        self.model_dir.mkdir(exist_ok=True)
    
    def download_data(self, symbol="ES=F", period="2y"):
        """Download ES futures data"""
        print(f"📊 Downloading {period} of {symbol} data...")
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval="1h")
            
            if data.empty:
                print("No ES=F data, trying SPY...")
                ticker = yf.Ticker("SPY")
                data = ticker.history(period=period, interval="1h")
            
            print(f"✅ Downloaded {len(data)} bars")
            return data
            
        except Exception as e:
            print(f"❌ Error downloading data: {e}")
            return None
    
    def engineer_features(self, data):
        """Create comprehensive features for transformer"""
        print("⚙️ Engineering features for transformer...")
        
        df = data.copy()
        features = []
        
        # Basic OHLCV
        features.extend(['Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Price ratios and changes
        df['Returns'] = df['Close'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Open_Close_Ratio'] = df['Open'] / df['Close']
        df['Volume_Change'] = df['Volume'].pct_change()
        features.extend(['Returns', 'High_Low_Ratio', 'Open_Close_Ratio', 'Volume_Change'])
        
        # Moving averages and ratios
        for period in [5, 10, 20, 50]:
            col_sma = f'SMA_{period}'
            col_ratio = f'Price_SMA_Ratio_{period}'
            
            df[col_sma] = df['Close'].rolling(period).mean()
            # Avoid division by zero
            df[col_ratio] = np.where(df[col_sma] != 0, df['Close'] / df[col_sma], 1.0)
            features.extend([col_ratio])  # Use ratio instead of absolute SMA
        
        # Technical indicators
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        # Avoid division by zero
        rs = np.where(loss != 0, gain / loss, 0)
        df['RSI'] = 100 - (100 / (1 + rs))
        features.append('RSI')
        
        # Bollinger Bands position
        bb_sma = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        bb_upper = bb_sma + (bb_std * 2)
        bb_lower = bb_sma - (bb_std * 2)
        # Avoid division by zero
        bb_range = bb_upper - bb_lower
        df['BB_Position'] = np.where(bb_range != 0, (df['Close'] - bb_lower) / bb_range, 0.5)
        features.append('BB_Position')
        
        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        features.extend(['MACD', 'MACD_Signal', 'MACD_Histogram'])
        
        # Volatility
        df['Volatility'] = df['Returns'].rolling(20).std()
        features.append('Volatility')
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        # Avoid division by zero
        df['Volume_Ratio'] = np.where(df['Volume_SMA'] != 0, df['Volume'] / df['Volume_SMA'], 1.0)
        features.append('Volume_Ratio')
        
        # Time features (important for intraday patterns)
        df['Hour'] = df.index.hour
        df['Day_of_Week'] = df.index.dayofweek
        df['Is_Market_Hours'] = ((df['Hour'] >= 9) & (df['Hour'] <= 16)).astype(int)
        features.extend(['Hour', 'Day_of_Week', 'Is_Market_Hours'])
        
        self.feature_columns = features
        print(f"✅ Created {len(features)} features for transformer")
        
        # Clean up infinite and NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with appropriate strategies
        for col in features:
            if col in df.columns:
                if df[col].dtype in ['float64', 'float32']:
                    # For numeric columns, fill with median
                    df[col] = df[col].fillna(df[col].median())
                else:
                    # For other columns, fill with mode or 0
                    df[col] = df[col].fillna(0)
        
        print(f"🧹 Cleaned data: {df.isnull().sum().sum()} NaN values remaining")
        
        return df
    
    def create_labels(self, data, forward_periods=6, threshold=0.001):
        """Create trading labels"""
        print(f"🏷️ Creating labels (forward_periods={forward_periods}, threshold={threshold})...")
        
        df = data.copy()
        
        # Calculate future returns
        df['Future_Return'] = df['Close'].shift(-forward_periods) / df['Close'] - 1
        
        # Create labels: 0=SELL, 1=HOLD, 2=BUY
        df['Label'] = 1  # Default HOLD
        df.loc[df['Future_Return'] > threshold, 'Label'] = 2   # BUY
        df.loc[df['Future_Return'] < -threshold, 'Label'] = 0  # SELL
        
        # Remove future data leakage
        df = df[:-forward_periods]
        
        # Print label distribution
        label_counts = df['Label'].value_counts().sort_index()
        total = len(df)
        print(f"📊 Label distribution:")
        print(f"   SELL: {label_counts.get(0, 0)} ({label_counts.get(0, 0)/total*100:.1f}%)")
        print(f"   HOLD: {label_counts.get(1, 0)} ({label_counts.get(1, 0)/total*100:.1f}%)")
        print(f"   BUY:  {label_counts.get(2, 0)} ({label_counts.get(2, 0)/total*100:.1f}%)")
        
        return df
    
    def create_sequences(self, data):
        """Create sequences for transformer training"""
        print(f"📦 Creating sequences (length={self.seq_length})...")
        
        # Prepare feature matrix
        feature_data = data[self.feature_columns].values
        labels = data['Label'].values
        
        # Scale features
        feature_data_scaled = self.scaler.fit_transform(feature_data)
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(self.seq_length, len(feature_data_scaled)):
            # Get sequence
            seq = feature_data_scaled[i-self.seq_length:i]
            sequences.append(seq)
            
            # Get target
            target = labels[i]
            targets.append(target)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        print(f"✅ Created {len(sequences)} sequences")
        print(f"   Sequence shape: {sequences.shape}")
        print(f"   Target shape: {targets.shape}")
        
        return sequences, targets
    
    def build_model(self):
        """Build transformer model"""
        print("🏗️ Building transformer model...")
        
        self.model = TradingTransformer(
            input_dim=len(self.feature_columns),
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            seq_length=self.seq_length,
            num_classes=3,  # SELL, HOLD, BUY
            dropout=0.1
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"📊 Model built:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        return self.model
    
    def train_model(self, sequences, targets):
        """Train the transformer model"""
        print("🚀 Starting transformer training...")
        
        # Create datasets
        # Use time series split for validation
        split_idx = int(len(sequences) * 0.8)
        
        train_sequences = sequences[:split_idx]
        train_targets = targets[:split_idx]
        val_sequences = sequences[split_idx:]
        val_targets = targets[split_idx:]
        
        train_dataset = ESTransformerDataset(train_sequences, train_targets)
        val_dataset = ESTransformerDataset(val_sequences, val_targets)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)  # Don't shuffle time series
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        # Training loop
        best_val_acc = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                signals, confidence = self.model(data)
                loss = criterion(signals, targets)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = signals.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, targets in val_loader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    signals, confidence = self.model(data)
                    loss = criterion(signals, targets)
                    
                    val_loss += loss.item()
                    _, predicted = signals.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            # Calculate metrics
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            scheduler.step()
            
            # Print progress
            if epoch % 5 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch+1:3d}/{self.epochs}: "
                      f"Train Loss: {train_loss/len(train_loader):.4f}, "
                      f"Train Acc: {train_acc:.2f}%, "
                      f"Val Loss: {val_loss/len(val_loader):.4f}, "
                      f"Val Acc: {val_acc:.2f}%")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                self.save_model(suffix="best")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
        
        print(f"✅ Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        
        # Load best model
        self.load_model(suffix="best")
        
        return best_val_acc
    
    def evaluate_model(self, sequences, targets):
        """Evaluate model performance"""
        print("📊 Evaluating transformer model...")
        
        # Use last 20% for testing
        split_idx = int(len(sequences) * 0.8)
        test_sequences = sequences[split_idx:]
        test_targets = targets[split_idx:]
        
        test_dataset = ESTransformerDataset(test_sequences, test_targets)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_confidences = []
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                signals, confidence = self.model(data)
                
                probs = torch.softmax(signals, dim=1)
                predicted = probs.argmax(dim=1)
                max_prob = probs.max(dim=1)[0]
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_confidences.extend((max_prob * confidence.squeeze()).cpu().numpy())
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
        
        # High confidence performance
        high_conf_mask = np.array(all_confidences) > 0.7
        if high_conf_mask.sum() > 0:
            high_conf_acc = np.mean(np.array(all_predictions)[high_conf_mask] == np.array(all_targets)[high_conf_mask])
            high_conf_count = high_conf_mask.sum()
        else:
            high_conf_acc = 0
            high_conf_count = 0
        
        print(f"📊 Transformer Performance:")
        print(f"   Overall Accuracy: {accuracy*100:.2f}%")
        print(f"   High Confidence (>70%) Accuracy: {high_conf_acc*100:.2f}% ({high_conf_count} samples)")
        
        # Class-wise performance
        from sklearn.metrics import classification_report
        class_names = ['SELL', 'HOLD', 'BUY']
        print("\n📊 Detailed Classification Report:")
        print(classification_report(all_targets, all_predictions, target_names=class_names))
        
        return accuracy
    
    def save_model(self, suffix=""):
        """Save the trained transformer model"""
        print("💾 Saving transformer model...")
        
        model_name = f"es_transformer_model{('_' + suffix) if suffix else ''}.pt"
        model_path = self.model_dir / model_name
        
        # Save complete model state
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': len(self.feature_columns),
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_layers': self.num_layers,
                'seq_length': self.seq_length
            },
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'device': str(self.device),
            'created_date': datetime.now().isoformat()
        }
        
        torch.save(save_dict, model_path)
        
        # Also save in joblib format for compatibility
        joblib_path = self.model_dir / f"es_transformer_complete{('_' + suffix) if suffix else ''}.joblib"
        joblib.dump(save_dict, joblib_path)
        
        print(f"✅ Model saved to {model_path}")
        return model_path
    
    def load_model(self, suffix=""):
        """Load a saved transformer model"""
        model_name = f"es_transformer_model{('_' + suffix) if suffix else ''}.pt"
        model_path = self.model_dir / model_name
        
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Rebuild model
            config = checkpoint['model_config']
            self.model = TradingTransformer(**config).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load other components
            self.scaler = checkpoint['scaler']
            self.feature_columns = checkpoint['feature_columns']
            
            print(f"✅ Model loaded from {model_path}")
            return True
        else:
            print(f"❌ Model file not found: {model_path}")
            return False
    
    def run_complete_training(self):
        """Run the complete transformer training pipeline"""
        print("🚀 ES Transformer Training Pipeline")
        print("="*60)
        
        try:
            # Step 1: Download data
            data = self.download_data()
            if data is None:
                return False
            
            # Step 2: Engineer features
            data_with_features = self.engineer_features(data)
            
            # Step 3: Create labels
            labeled_data = self.create_labels(data_with_features)
            
            # Step 4: Create sequences
            sequences, targets = self.create_sequences(labeled_data)
            
            # Step 5: Build model
            self.build_model()
            
            # Step 6: Train model
            best_acc = self.train_model(sequences, targets)
            
            # Step 7: Evaluate model
            final_acc = self.evaluate_model(sequences, targets)
            
            # Step 8: Save final model
            self.save_model()
            
            print("="*60)
            print("🎉 Transformer training completed successfully!")
            print(f"📊 Final accuracy: {final_acc*100:.2f}%")
            print("🔥 Your transformer model is ready for trading!")
            print("="*60)
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    print("🚀 ES Transformer Model Training")
    print("Advanced deep learning for futures trading")
    print("="*50)
    
    # Run training
    trainer = ESTransformerTrainer()
    success = trainer.run_complete_training()
    
    if success:
        print("✅ Transformer model ready!")
        print("🔄 Restart NinjaTrader to use the new model")
    else:
        print("❌ Training failed")