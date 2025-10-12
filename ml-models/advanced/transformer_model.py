"""
ES Trading Transformer Model
Advanced deep learning transformer for sequence-to-sequence trading predictions
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import math

class PositionalEncoding(nn.Module):
    """
    Positional encoding for time series data
    Critical for transformers to understand sequence order
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TradingTransformer(nn.Module):
    """
    Transformer model for ES futures trading signals
    
    Architecture:
    - Input: Sequence of market data (price, volume, indicators)
    - Transformer Encoder: Learns temporal patterns and relationships
    - Output: Trading signal probabilities (BUY/SELL/HOLD)
    """
    
    def __init__(self, 
                 input_dim=20,      # Number of features per timestep
                 d_model=256,       # Transformer embedding dimension
                 nhead=8,           # Number of attention heads
                 num_layers=6,      # Number of transformer layers
                 seq_length=100,    # Sequence length (100 bars = ~1.5 hours)
                 num_classes=3,     # BUY, SELL, HOLD
                 dropout=0.1):
        
        super(TradingTransformer, self).__init__()
        
        self.d_model = d_model
        self.seq_length = seq_length
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Confidence head (for risk management)
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, mask=None):
        """
        Forward pass
        
        Args:
            x: (batch_size, seq_length, input_dim)
            mask: Optional attention mask
            
        Returns:
            signals: (batch_size, num_classes) - Trading signal probabilities
            confidence: (batch_size, 1) - Prediction confidence
        """
        batch_size, seq_length, _ = x.shape
        
        # Project input to model dimension
        x = self.input_projection(x)  # (batch, seq, d_model)
        x = x * math.sqrt(self.d_model)  # Scale embeddings
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq, batch, d_model) - transformer expects this
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Use the last timestep for prediction (could also use attention pooling)
        last_hidden = encoded[-1]  # (batch, d_model)
        
        # Generate trading signals
        signals = self.classifier(last_hidden)  # (batch, num_classes)
        
        # Generate confidence score
        confidence = self.confidence_head(last_hidden)  # (batch, 1)
        
        return signals, confidence

class ESTransformerTrainer:
    """
    Trainer for the ES Trading Transformer
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
    
    def create_sequences(self, data, seq_length=100, target_col='label'):
        """
        Create sequences for transformer training
        
        Each sequence: [t-99, t-98, ..., t-1, t] -> predict t+1
        """
        sequences = []
        targets = []
        
        feature_cols = [col for col in data.columns if col != target_col]
        
        for i in range(seq_length, len(data)):
            # Get sequence of features
            seq = data[feature_cols].iloc[i-seq_length:i].values
            sequences.append(seq)
            
            # Get target (future signal)
            target = data[target_col].iloc[i]
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            signals, confidence = self.model(data)
            loss = self.criterion(signals, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = signals.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, accuracy
    
    def predict_signal(self, market_data_sequence):
        """
        Generate trading signal from market data sequence
        
        Args:
            market_data_sequence: Recent market data (seq_length timesteps)
            
        Returns:
            signal: 'BUY'/'SELL'/'HOLD'
            confidence: 0-1 confidence score
        """
        self.model.eval()
        
        with torch.no_grad():
            # Convert to tensor
            x = torch.FloatTensor(market_data_sequence).unsqueeze(0).to(self.device)
            
            # Get prediction
            signals, confidence = self.model(x)
            
            # Convert to probabilities
            probs = torch.softmax(signals, dim=1)
            predicted_class = probs.argmax().item()
            max_prob = probs.max().item()
            confidence_score = confidence.item()
            
            # Map to signal names
            signal_map = ['SELL', 'HOLD', 'BUY']
            signal = signal_map[predicted_class]
            
            # Combine prediction confidence with model confidence
            final_confidence = max_prob * confidence_score
            
            return signal, final_confidence

def create_transformer_features(price_data):
    """
    Create rich feature set for transformer model
    """
    features = []
    
    # Price features
    features.extend([
        price_data['Open'], price_data['High'], 
        price_data['Low'], price_data['Close'], price_data['Volume']
    ])
    
    # Returns and ratios
    returns = price_data['Close'].pct_change()
    features.extend([
        returns,
        price_data['High'] / price_data['Low'],
        price_data['Close'] / price_data['Open'],
        price_data['Volume'].pct_change()
    ])
    
    # Moving averages
    for period in [5, 10, 20, 50]:
        sma = price_data['Close'].rolling(period).mean()
        features.extend([
            sma,
            price_data['Close'] / sma
        ])
    
    # Technical indicators
    # RSI
    delta = price_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)
    
    # Bollinger Bands
    bb_sma = price_data['Close'].rolling(20).mean()
    bb_std = price_data['Close'].rolling(20).std()
    bb_upper = bb_sma + (bb_std * 2)
    bb_lower = bb_sma - (bb_std * 2)
    bb_position = (price_data['Close'] - bb_lower) / (bb_upper - bb_lower)
    features.append(bb_position)
    
    return pd.concat(features, axis=1)

# Example usage for ES futures
if __name__ == "__main__":
    print("🤖 ES Trading Transformer Model")
    print("Advanced deep learning for futures trading")
    
    # Model configuration
    model = TradingTransformer(
        input_dim=20,      # 20 features per timestep
        d_model=256,       # Model dimension
        nhead=8,           # 8 attention heads
        num_layers=6,      # 6 transformer layers
        seq_length=100,    # 100-bar sequences (~1.5 hours)
        num_classes=3      # BUY/SELL/HOLD
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Ready for training on ES futures data!")