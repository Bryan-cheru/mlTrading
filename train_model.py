"""
ES ML Trading System - Model Training Script
Run this to train your ML models for ES futures trading
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'ml-models'))

# Import the model trainer - fix path with hyphens
import importlib.util
trainer_path = os.path.join(project_root, 'ml-models', 'training', 'es_model_trainer.py')
spec = importlib.util.spec_from_file_location("es_model_trainer", trainer_path)
trainer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer_module)
ESMLModelTrainer = trainer_module.ESMLModelTrainer
import pandas as pd

def main():
    """
    Main training script - runs complete ML pipeline
    """
    print("🚀 ES ML Trading System - Model Training")
    print("=" * 60)
    print("This will train ML models for ES futures trading")
    print("Training data: 2 years of ES futures historical data")
    print("Model type: Random Forest with institutional features")
    print("=" * 60)
    
    # Check if user wants to proceed
    response = input("\n🤔 Start training? (y/n): ").strip().lower()
    if response != 'y':
        print("❌ Training cancelled")
        return
    
    try:
        # Initialize trainer and run complete pipeline
        print("🔧 Initializing ES ML Trainer...")
        trainer = ESMLModelTrainer()
        
        # Download training data
        data = trainer.download_training_data(symbol="ES=F", period="2y")
        if data is None:
            print("❌ Failed to download training data")
            return
        
        # Engineer features
        print("⚙️ Engineering features...")
        featured_data = trainer.engineer_features(data)
        
        # Create labels
        print("🎯 Creating trading labels...")
        labeled_data = trainer.create_labels(featured_data, forward_periods=6, threshold=0.001)
        
        # Prepare training data
        print("📊 Preparing training data...")
        X, y = trainer.prepare_training_data(labeled_data)
        
        # Train model
        print("🤖 Training ML model...")
        cv_scores = trainer.train_model(X, y, n_splits=5)
        
        # Evaluate model
        print("📈 Evaluating model performance...")
        trainer.evaluate_model(X, y)
        
        # Save model
        print("💾 Saving trained model...")
        trainer.save_model()
        
        if trainer.is_trained:
            print("\n🎉 SUCCESS! Your ML model is ready for trading!")
            print("\n📋 Next Steps:")
            print("1. 🔄 Restart NinjaTrader 8")
            print("2. 🎮 Open ES ML Trading System from Tools menu")
            print("3. ▶️ Click 'Start System' to begin ML-powered trading")
            print("4. 📊 Monitor signals in the professional interface")
            
            print(f"\n💾 Model saved to: models/es_ml_model.joblib")
            print("🔧 Model will automatically be used by NinjaTrader AddOn")
            
        else:
            print("\n❌ Training failed. Check error messages above.")
            
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        print("🔄 Falling back to technical analysis mode")

def quick_test():
    """
    Quick test of the training pipeline with minimal data
    """
    print("🧪 Quick Test Mode - Training with 6 months data")
    print("=" * 50)
    
    trainer = ESMLModelTrainer()
    
    # Download smaller dataset
    data = trainer.download_training_data(symbol="ES=F", period="6mo")
    if data is None:
        print("❌ Could not download test data")
        return
    
    # Quick training pipeline
    featured_data = trainer.engineer_features(data)
    labeled_data = trainer.create_labels(featured_data, forward_periods=3, threshold=0.0005)
    X, y = trainer.prepare_training_data(labeled_data)
    
    # Train with fewer cross-validation splits
    cv_scores = trainer.train_model(X, y, n_splits=3)
    trainer.evaluate_model(X, y)
    
    print("\n✅ Quick test completed!")
    print("🔧 Use 'main()' for full training with 2 years of data")

def check_requirements():
    """
    Check if all required packages are installed
    """
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'yfinance', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print(f"\n📦 Install with: pip install {' '.join(missing_packages)}")
        return False
    else:
        print("✅ All required packages are installed")
        return True

if __name__ == "__main__":
    print("🎯 ES ML Trading System - Training Interface")
    print("=" * 50)
    
    # Check requirements first
    if not check_requirements():
        print("\n🛠️ Please install missing packages and try again")
        sys.exit(1)
    
    print("\nChoose training mode:")
    print("1. 🚀 Full Training (2 years data, production model)")
    print("2. 🧪 Quick Test (6 months data, testing)")
    print("3. ❌ Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        quick_test()
    elif choice == "3":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")