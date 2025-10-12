"""
GPU Environment Setup for High-Performance ES Transformer Training
Optimizes your powerful hardware for maximum training speed
"""

import subprocess
import sys
import os
import torch

def install_packages():
    """Install all required packages for GPU training"""
    packages = [
        "torch",
        "torchvision", 
        "torchaudio",
        "--index-url https://download.pytorch.org/whl/cu121",  # CUDA 12.1
        "transformers",
        "accelerate",
        "datasets",
        "matplotlib",
        "seaborn",
        "plotly",
        "tqdm",
        "tensorboard",
        "wandb",  # For experiment tracking
        "optuna",  # For hyperparameter optimization
    ]
    
    print("🔧 Installing GPU-optimized packages...")
    for package in packages:
        if package.startswith("--"):
            continue
        print(f"📦 Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("✅ All packages installed!")

def check_gpu_setup():
    """Check GPU configuration and optimization"""
    print("🔍 Checking GPU setup...")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.version.cuda}")
        print(f"🔥 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"🚀 GPU Count: {torch.cuda.device_count()}")
        
        # Test GPU performance
        print("🧪 Testing GPU performance...")
        device = torch.device('cuda')
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        
        import time
        start = time.time()
        for _ in range(100):
            z = torch.mm(x, y)
        torch.cuda.synchronize()
        end = time.time()
        
        print(f"⚡ GPU Performance: {(end - start) * 1000:.2f}ms for 100 matrix multiplications")
        print("✅ GPU setup verified!")
        
        return True
    else:
        print("❌ CUDA not available!")
        print("💡 Make sure you have:")
        print("   - NVIDIA GPU with CUDA support")
        print("   - CUDA drivers installed")
        print("   - PyTorch with CUDA support")
        return False

def optimize_environment():
    """Set environment variables for optimal performance"""
    print("⚙️ Optimizing environment for your hardware...")
    
    # Set environment variables for maximum performance
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async CUDA calls
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # Use cuDNN v8 API
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # Optimize memory
    
    # Set number of threads to use your RAM effectively
    torch.set_num_threads(16)  # Adjust based on your CPU cores
    
    print("✅ Environment optimized!")
    
    return {
        'batch_size': 64,  # Large batch for your GPU
        'num_workers': 8,  # Data loading workers
        'pin_memory': True,  # Faster GPU transfers
        'persistent_workers': True,  # Keep workers alive
    }

def create_training_config():
    """Create optimized training configuration"""
    config = {
        'model': {
            'd_model': 768,        # Large model for powerful GPU
            'nhead': 24,           # More attention heads
            'num_layers': 16,      # Deep network
            'seq_length': 300,     # Longer sequences (12+ hours)
            'dropout': 0.1,
        },
        'training': {
            'batch_size': 64,      # Large batches
            'learning_rate': 2e-4,
            'epochs': 200,
            'warmup_steps': 1000,
            'gradient_clip': 1.0,
            'mixed_precision': True,
        },
        'data': {
            'sequence_length': 300,
            'num_workers': 8,
            'pin_memory': True,
            'prefetch_factor': 4,
        },
        'optimization': {
            'use_flash_attention': True,  # If available
            'compile_model': True,        # PyTorch 2.0 optimization
            'use_channels_last': True,    # Memory layout optimization
        }
    }
    
    # Save config
    import json
    with open('gpu_training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("📄 Training configuration saved to gpu_training_config.json")
    return config

def create_monitoring_script():
    """Create GPU monitoring script"""
    monitoring_script = '''
import psutil
import GPUtil
import time
import pandas as pd
from datetime import datetime

def monitor_system():
    """Monitor system resources during training"""
    print("📊 System Monitoring Started")
    print("=" * 50)
    
    while True:
        # GPU monitoring
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            print(f"🔥 GPU Usage: {gpu.load*100:.1f}%")
            print(f"💾 GPU Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")
            print(f"🌡️ GPU Temp: {gpu.temperature}°C")
        
        # CPU monitoring
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        print(f"🖥️ CPU Usage: {cpu_percent:.1f}%")
        print(f"💽 RAM Usage: {memory.used/1e9:.1f}GB / {memory.total/1e9:.1f}GB ({memory.percent:.1f}%)")
        print(f"📅 Time: {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 50)
        
        time.sleep(10)  # Update every 10 seconds

if __name__ == "__main__":
    monitor_system()
'''
    
    with open('monitor_training.py', 'w') as f:
        f.write(monitoring_script)
    
    print("📊 Monitoring script created: monitor_training.py")

def main():
    """Complete setup for high-performance training"""
    print("🚀 High-Performance ES Transformer Setup")
    print("=" * 60)
    print("💪 Optimizing for your powerful hardware:")
    print("   - GPU acceleration")
    print("   - 128GB+ RAM utilization") 
    print("   - Multi-core CPU optimization")
    print("=" * 60)
    
    # Check current setup
    gpu_available = check_gpu_setup()
    
    if not gpu_available:
        print("⚠️ GPU not detected. Training will be slower on CPU.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Install packages
    try:
        install_packages()
    except Exception as e:
        print(f"⚠️ Package installation warning: {e}")
    
    # Optimize environment
    config = optimize_environment()
    
    # Create training config
    training_config = create_training_config()
    
    # Create monitoring
    create_monitoring_script()
    
    print("\n🎉 Setup Complete!")
    print("=" * 50)
    print("📋 Next Steps:")
    print("1. 🚀 Run: python ml-models/training/gpu_transformer_trainer.py")
    print("2. 📊 Monitor: python monitor_training.py (in separate terminal)")
    print("3. ⏱️ Expected training time: 2-4 hours on your GPU")
    print("4. 🎯 Expected final accuracy: 75-85%")
    print("\n💡 Pro Tips:")
    print("- Keep GPU temperature below 80°C")
    print("- Monitor RAM usage (should use 20-40GB during training)")
    print("- Training will auto-save best model")
    print("- Use TensorBoard for real-time metrics")

if __name__ == "__main__":
    main()