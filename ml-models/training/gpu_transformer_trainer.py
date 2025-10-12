"""
High-Performance GPU-Optimized ES Transformer Training
Designed for powerful hardware with GPU acceleration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import math
import warnings
import os
from pathlib import Path
import json
import pickle
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class HighPerformanceESDataset(Dataset):
    """Optimized dataset for GPU training"""
    
    def __init__(self, sequences, targets, device='cuda'):
        self.sequences = torch.FloatTensor(sequences).to(device)
        self.targets = torch.LongTensor(targets).to(device)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class OptimizedPositionalEncoding(nn.Module):
    """GPU-optimized positional encoding"""
    
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class HighPerformanceESTransformer(nn.Module):
    """
    GPU-Optimized Transformer for ES Futures Trading
    Designed for maximum performance on powerful hardware
    """
    
    def __init__(self, 
                 input_dim=25,
                 d_model=512,          # Larger model for your powerful GPU
                 nhead=16,             # More attention heads
                 num_layers=12,        # Deeper network
                 seq_length=200,       # Longer sequences (3+ hours of data)
                 num_classes=3,
                 dropout=0.1,
                 max_len=1000):
        super().__init__()
        
        self.d_model = d_model
        self.seq_length = seq_length
        
        # Input embedding with layer norm
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoder = OptimizedPositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder with optimizations
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,  # Better for GPU performance
            norm_first=True    # Pre-norm for better training
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Multi-head attention pooling for better sequence aggregation
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Enhanced classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Xavier initialization for better convergence"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Input embedding
        x = self.input_embedding(x)  # (batch, seq, d_model)
        
        # Positional encoding
        x = x.transpose(0, 1)  # (seq, batch, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch, seq, d_model)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Attention pooling - learns to focus on important timesteps
        query = encoded.mean(dim=1, keepdim=True)  # (batch, 1, d_model)
        pooled, _ = self.attention_pool(query, encoded, encoded)
        pooled = pooled.squeeze(1)  # (batch, d_model)
        
        # Predictions
        logits = self.classifier(pooled)
        confidence = self.confidence_head(pooled)
        
        return logits, confidence

class HighPerformanceTrainer:
    """GPU-optimized trainer with advanced features"""
    
    def __init__(self, model, device='cuda', learning_rate=1e-4):
        self.model = model.to(device)
        self.device = device
        
        # Advanced optimizer with gradient clipping
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.95)  # Better for transformers
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate * 10,
            epochs=100,
            steps_per_epoch=100,  # Will update with actual steps
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Mixed precision training for speed
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0
        
    def train_epoch(self, train_loader, epoch):
        """Train one epoch with mixed precision"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, (data, targets) in enumerate(pbar):
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                logits, confidence = self.model(data)
                loss = self.criterion(logits, targets)
                
                # Add confidence regularization
                conf_loss = torch.mean((confidence - 0.5) ** 2) * 0.1
                total_batch_loss = loss + conf_loss
            
            # Mixed precision backward pass
            self.scaler.scale(total_batch_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'LR': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validation with confidence analysis"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        high_conf_correct = 0
        high_conf_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                with torch.cuda.amp.autocast():
                    logits, confidence = self.model(data)
                    loss = self.criterion(logits, targets)
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # High confidence predictions (>0.8)
                high_conf_mask = confidence.squeeze() > 0.8
                if high_conf_mask.sum() > 0:
                    high_conf_correct += predicted[high_conf_mask].eq(targets[high_conf_mask]).sum().item()
                    high_conf_total += high_conf_mask.sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(val_loader)
        high_conf_acc = 100. * high_conf_correct / high_conf_total if high_conf_total > 0 else 0
        
        return avg_loss, accuracy, high_conf_acc, high_conf_total
    
    def train_model(self, train_loader, val_loader, epochs=100, save_path="models/"):
        """Complete training loop"""
        print(f"🚀 Training on {self.device} with mixed precision")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        Path(save_path).mkdir(exist_ok=True)
        best_model_path = Path(save_path) / "es_transformer_best.pt"
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_acc, high_conf_acc, high_conf_count = self.validate(val_loader)
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'train_acc': train_acc,
                    'high_conf_acc': high_conf_acc
                }, best_model_path)
                print(f"💾 New best model saved! Val Acc: {val_acc:.2f}%")
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:3d}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                      f"High Conf Acc: {high_conf_acc:.2f}% ({high_conf_count} samples)")
        
        print(f"✅ Training completed! Best validation accuracy: {self.best_val_acc:.2f}%")
        return best_model_path

def create_advanced_features(data):
    """Create comprehensive feature set for transformer"""
    df = data.copy()
    
    # Price features
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['high_low_ratio'] = df['High'] / df['Low']
    df['open_close_ratio'] = df['Open'] / df['Close']
    
    # Volume features
    df['volume_change'] = df['Volume'].pct_change()
    df['volume_ma'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma']
    
    # Moving averages and ratios
    periods = [5, 10, 20, 50, 100]
    for period in periods:
        sma = df['Close'].rolling(period).mean()
        df[f'sma_{period}'] = sma
        df[f'price_sma_ratio_{period}'] = df['Close'] / sma
    
    # SMA ratios
    df['sma_ratio_5_20'] = df['sma_5'] / df['sma_20']
    df['sma_ratio_20_50'] = df['sma_20'] / df['sma_50']
    
    # Volatility
    for period in [10, 20, 50]:
        df[f'volatility_{period}'] = df['returns'].rolling(period).std()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    bb_sma = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['bb_upper'] = bb_sma + (bb_std * 2)
    df['bb_lower'] = bb_sma - (bb_std * 2)
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Momentum indicators
    df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
    df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
    
    # Time features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_market_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 16)).astype(int)
    
    # Drop original OHLCV to keep only features
    feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
    
    return df[feature_cols]

def main():
    """High-performance training pipeline with config support"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU-Optimized ES Transformer Training')
    parser.add_argument('--config', type=str, help='Path to training configuration JSON')
    args = parser.parse_args()
    
    # Load config if provided
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"� Loaded config from: {args.config}")
    
    print("�🚀 High-Performance GPU ES Transformer Training")
    print("=" * 60)
    
    # Device setup
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"🔥 Using GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = 'cpu'
        print("⚠️ CUDA not available, using CPU")
    
    # Data loading with config
    data_period = config.get('data', {}).get('data_period', '5y')
    print(f"📊 Loading {data_period} of ES futures data...")
    ticker = yf.Ticker("ES=F")
    data = ticker.history(period=data_period, interval="1h")
    print(f"✅ Loaded {len(data)} bars")
    
    # Feature engineering
    print("⚙️ Creating advanced features...")
    features = create_advanced_features(data)
    
    # Create labels
    print("🏷️ Creating trading labels...")
    returns = data['Close'].pct_change(6)  # 6-hour forward returns
    threshold = 0.002  # 0.2% threshold
    
    labels = pd.Series(1, index=returns.index)  # Default HOLD
    labels[returns > threshold] = 2   # BUY
    labels[returns < -threshold] = 0  # SELL
    
    # Combine features and labels
    combined = pd.concat([features, labels.rename('label')], axis=1).dropna()
    print(f"📊 Final dataset: {len(combined)} samples, {len(features.columns)} features")
    
    # Label distribution
    label_counts = combined['label'].value_counts().sort_index()
    print(f"📈 Label distribution:")
    print(f"   SELL: {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(combined)*100:.1f}%)")
    print(f"   HOLD: {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(combined)*100:.1f}%)")
    print(f"   BUY:  {label_counts.get(2, 0)} ({label_counts.get(2, 0)/len(combined)*100:.1f}%)")
    
    # Create sequences with config
    model_config = config.get('model', {})
    seq_length = model_config.get('seq_length', 200)
    print(f"📦 Creating sequences (length={seq_length})...")
    
    sequences, targets = [], []
    feature_data = combined.drop('label', axis=1).values
    label_data = combined['label'].values
    
    for i in range(seq_length, len(combined)):
        sequences.append(feature_data[i-seq_length:i])
        targets.append(label_data[i])
    
    sequences = np.array(sequences, dtype=np.float32)
    targets = np.array(targets, dtype=np.int64)
    
    print(f"✅ Created {len(sequences)} sequences")
    print(f"   Shape: {sequences.shape}")
    
    # Train/validation/test split
    data_config = config.get('data', {})
    val_split = data_config.get('validation_split', 0.2)
    test_split = data_config.get('test_split', 0.1)
    
    # Calculate split indices
    total_len = len(sequences)
    test_idx = int(total_len * (1 - test_split))
    val_idx = int(test_idx * (1 - val_split))
    
    train_seq = sequences[:val_idx]
    val_seq = sequences[val_idx:test_idx]
    test_seq = sequences[test_idx:]
    
    train_targets = targets[:val_idx]
    val_targets = targets[val_idx:test_idx]
    test_targets = targets[test_idx:]
    
    print(f"📊 Data splits:")
    print(f"   Train: {len(train_seq)} samples")
    print(f"   Validation: {len(val_seq)} samples")
    print(f"   Test: {len(test_seq)} samples")
    
    # Create datasets
    train_dataset = HighPerformanceESDataset(train_seq, train_targets, device)
    val_dataset = HighPerformanceESDataset(val_seq, val_targets, device)
    test_dataset = HighPerformanceESDataset(test_seq, test_targets, device)
    
    # Create data loaders with config
    training_config = config.get('training', {})
    batch_size = training_config.get('batch_size', 32 if device == 'cuda' else 8)
    num_workers = data_config.get('num_workers', 4)
    pin_memory = data_config.get('pin_memory', True)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Create model with config
    print("🏗️ Building high-performance transformer...")
    model = HighPerformanceESTransformer(
        input_dim=sequences.shape[2],
        d_model=model_config.get('d_model', 512),
        nhead=model_config.get('nhead', 16),
        num_layers=model_config.get('num_layers', 12),
        seq_length=seq_length,
        num_classes=model_config.get('num_classes', 3),
        dropout=model_config.get('dropout', 0.1)
    )
    
    # Compile model for PyTorch 2.0+ optimization
    if hasattr(torch, 'compile') and config.get('optimization', {}).get('compile_model', True):
        print("⚡ Compiling model for optimization...")
        model = torch.compile(model)
    
    # Create trainer with config
    learning_rate = training_config.get('learning_rate', 1e-4)
    trainer = HighPerformanceTrainer(model, device=device, learning_rate=learning_rate)
    
    # Update trainer settings from config
    trainer.scheduler.steps_per_epoch = len(train_loader)
    
    # Setup progress tracking
    progress_file = 'training_progress.json'
    
    def save_progress(epoch, train_loss, train_acc, val_loss, val_acc):
        """Save training progress for monitoring"""
        try:
            if os.path.exists(progress_file):
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
            else:
                progress = {'epochs': [], 'train_losses': [], 'val_losses': [], 
                           'train_accs': [], 'val_accs': []}
            
            progress['epochs'].append(epoch)
            progress['train_losses'].append(train_loss)
            progress['val_losses'].append(val_loss)
            progress['train_accs'].append(train_acc)
            progress['val_accs'].append(val_acc)
            
            with open(progress_file, 'w') as f:
                json.dump(progress, f)
        except:
            pass
    
    # Train model with config
    epochs = training_config.get('epochs', 100)
    print(f"🚀 Starting high-performance training ({epochs} epochs)...")
    print("💡 Monitor with: python training_dashboard.py")
    
    # Custom training loop with progress tracking
    best_val_acc = 0.0
    patience = training_config.get('early_stopping_patience', 15)
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
        
        # Validation
        val_loss, val_acc, high_conf_acc, high_conf_count = trainer.validate(val_loader)
        
        # Save progress
        save_progress(epoch, train_loss, train_acc, val_loss, val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'high_conf_acc': high_conf_acc,
                'config': config
            }, 'models/es_transformer_best.pt')
            print(f"💾 New best model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                  f"High Conf Acc: {high_conf_acc:.2f}% ({high_conf_count} samples)")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"🛑 Early stopping triggered after {epoch+1} epochs")
            break
    
    # Final evaluation on test set
    print("\n🧪 Final evaluation on test set...")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Load best model
    checkpoint = torch.load('models/es_transformer_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_high_conf_acc, test_high_conf_count = trainer.validate(test_loader)
    
    print(f"\n🎯 Final Results:")
    print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"   Test Accuracy: {test_acc:.2f}%")
    print(f"   High Confidence Test Accuracy: {test_high_conf_acc:.2f}% ({test_high_conf_count} samples)")
    
    print(f"\n🎉 Training completed! Model saved to: models/es_transformer_best.pt")
    print("✅ Ready for integration with NinjaTrader AddOn!")

if __name__ == "__main__":
    main()