"""
GPU-Optimized Transformer Trainer for ES Futures
High-performance training on remote California PC
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import yfinance as yf
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import json
import warnings
warnings.filterwarnings('ignore')

class HighPerformanceESTransformer(nn.Module):
    """GPU-optimized transformer for ES futures trading"""
    
    def __init__(self, input_dim=50, d_model=512, nhead=16, num_layers=12, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_head = nn.Linear(d_model, 3)  # Buy, Sell, Hold
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        seq_len = x.size(1)
        x = self.input_projection(x)
        x = x + self.positional_encoding[:seq_len].unsqueeze(0)
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.output_head(x[:, -1])  # Use last timestep
        return x

class ESTransformerTrainer:
    """High-performance training coordinator"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def prepare_data(self):
        """Download and prepare ES futures data"""
        print("Downloading ES futures data...")
        
        # ES futures symbol
        es = yf.Ticker("ES=F")
        data = es.history(period="2y", interval="1h")
        
        if data.empty:
            print("No data found, using SPY as proxy...")
            spy = yf.Ticker("SPY")
            data = spy.history(period="2y", interval="1h")
        
        print(f"Downloaded {len(data)} data points")
        
        # Feature engineering
        features = self.create_features(data)
        return features
    
    def create_features(self, data):
        """Create comprehensive feature set"""
        features = pd.DataFrame()
        
        # Price features
        features['returns'] = data['Close'].pct_change()
        features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        features['high_low_ratio'] = data['High'] / data['Low']
        features['close_open_ratio'] = data['Close'] / data['Open']
        
        # Technical indicators
        for window in [5, 10, 20, 50]:
            features[f'sma_{window}'] = data['Close'].rolling(window).mean()
            features[f'std_{window}'] = data['Close'].rolling(window).std()
            features[f'rsi_{window}'] = self.calculate_rsi(data['Close'], window)
        
        # Volume features
        features['volume_sma'] = data['Volume'].rolling(20).mean()
        features['volume_ratio'] = data['Volume'] / features['volume_sma']
        
        # Volatility features
        features['volatility'] = features['returns'].rolling(20).std()
        features['volatility_ratio'] = features['volatility'] / features['volatility'].rolling(50).mean()
        
        # Time features
        features['hour'] = pd.to_datetime(data.index).hour
        features['day_of_week'] = pd.to_datetime(data.index).dayofweek
        features['month'] = pd.to_datetime(data.index).month
        
        # Create target
        features['target'] = np.where(
            features['returns'].shift(-1) > 0.001, 1,  # Buy
            np.where(features['returns'].shift(-1) < -0.001, 0, 2)  # Sell, Hold
        )
        
        return features.dropna()
    
    def calculate_rsi(self, prices, window):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def create_sequences(self, data, seq_length=100):
        """Create sequences for transformer training"""
        feature_cols = [col for col in data.columns if col != 'target']
        
        # Scale features
        scaler = RobustScaler()
        scaled_features = scaler.fit_transform(data[feature_cols])
        
        sequences = []
        targets = []
        
        for i in range(seq_length, len(data)):
            sequences.append(scaled_features[i-seq_length:i])
            targets.append(data['target'].iloc[i])
        
        return np.array(sequences), np.array(targets), scaler, feature_cols
    
    def train_model(self, sequences, targets, epochs=50, batch_size=32):
        """Train transformer model with mixed precision"""
        print(f"Training transformer with {len(sequences)} sequences...")
        
        # Convert to tensors
        X = torch.tensor(sequences, dtype=torch.float32).to(self.device)
        y = torch.tensor(targets, dtype=torch.long).to(self.device)
        
        # Initialize model
        model = HighPerformanceESTransformer(
            input_dim=sequences.shape[2],
            d_model=512,
            nhead=16,
            num_layers=12
        ).to(self.device)
        
        # Mixed precision training
        scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            # Batch training
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                optimizer.zero_grad()
                
                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            accuracy = 100 * correct / total
            avg_loss = total_loss / (len(X) // batch_size)
            scheduler.step(avg_loss)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        return model
    
    def save_model(self, model, scaler, feature_cols):
        """Save trained model and metadata"""
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Save model
        torch.save(model.state_dict(), models_dir / "es_transformer.pth")
        
        # Save scaler and metadata
        import pickle
        with open(models_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        
        metadata = {
            "feature_columns": feature_cols,
            "model_type": "transformer",
            "input_dim": len(feature_cols),
            "d_model": 512,
            "num_layers": 12,
            "nhead": 16
        }
        
        with open(models_dir / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print("Model saved successfully!")

def main():
    """Main training function"""
    print("ES Futures Transformer Training")
    print("="*50)
    
    trainer = ESTransformerTrainer()
    
    # Prepare data
    data = trainer.prepare_data()
    print(f"Features shape: {data.shape}")
    
    # Create sequences
    sequences, targets, scaler, feature_cols = trainer.create_sequences(data)
    print(f"Sequences shape: {sequences.shape}")
    
    # Train model
    model = trainer.train_model(sequences, targets)
    
    # Save model
    trainer.save_model(model, scaler, feature_cols)
    
    print("\nTraining completed successfully!")
    print("Model saved to: models/es_transformer.pth")

if __name__ == "__main__":
    main()