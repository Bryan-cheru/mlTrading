#!/usr/bin/env python3
"""
Advanced ML Ensemble Test Suite
Test the sophisticated LSTM + Transformer + XGBoost + LightGBM ensemble
Focus: <10ms inference latency, model performance, A/B testing
"""

import sys
import os
import asyncio
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'ml-models'))

from advanced_ensemble import AdvancedMLEnsemble, ModelConfig, ModelPerformanceMonitor

class MLEnsembleTestSuite:
    """Comprehensive test suite for Advanced ML Ensemble"""
    
    def __init__(self):
        self.ensemble = None
        self.test_data = None
        self.performance_results = {}
        
    def generate_realistic_market_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """Generate realistic synthetic market data for testing"""
        print("📊 Generating realistic market data for testing...")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Base parameters
        initial_price = 100.0
        
        # Generate timestamps
        start_date = datetime.now() - timedelta(days=n_samples//288)  # 5-minute bars
        timestamps = [start_date + timedelta(minutes=5*i) for i in range(n_samples)]
        
        # Generate realistic price movements
        returns = np.random.normal(0, 0.001, n_samples)  # 0.1% std
        
        # Add some trends and volatility clustering
        trend_changes = np.random.choice([-1, 0, 1], n_samples, p=[0.1, 0.8, 0.1])
        volatility_regime = np.random.choice([0.5, 1.0, 2.0], n_samples, p=[0.3, 0.5, 0.2])
        
        returns = returns * volatility_regime + trend_changes * 0.0005
        
        # Calculate prices
        prices = [initial_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate OHLCV data
        data = []
        for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
            # Generate realistic OHLC around close price
            spread = close * 0.002  # 0.2% spread
            
            high = close + np.random.uniform(0, spread)
            low = close - np.random.uniform(0, spread)
            open_price = low + np.random.uniform(0, high - low)
            
            # Ensure OHLC consistency
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            # Generate volume
            base_volume = 10000
            volume_multiplier = np.random.lognormal(0, 0.5)
            volume = int(base_volume * volume_multiplier)
            
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        
        print(f"✅ Generated {len(df):,} market data points")
        print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print(f"   Volume range: {df['volume'].min():,} - {df['volume'].max():,}")
        
        return df
    
    async def test_model_initialization(self):
        """Test model initialization and configuration"""
        print("\n" + "="*80)
        print("🧠 TESTING ML ENSEMBLE INITIALIZATION")
        print("="*80)
        
        try:
            # Test default configuration
            print("\n1️⃣ Testing default configuration...")
            config = ModelConfig()
            self.ensemble = AdvancedMLEnsemble(config)
            
            print(f"   ✅ LSTM config: hidden_size={config.lstm_hidden_size}, layers={config.lstm_num_layers}")
            print(f"   ✅ Transformer config: d_model={config.transformer_d_model}, heads={config.transformer_nhead}")
            print(f"   ✅ XGBoost config: n_estimators={config.xgb_n_estimators}, max_depth={config.xgb_max_depth}")
            print(f"   ✅ LightGBM config: n_estimators={config.lgb_n_estimators}, num_leaves={config.lgb_num_leaves}")
            print(f"   ✅ Ensemble weights: {config.ensemble_weights}")
            
            # Test custom configuration
            print("\n2️⃣ Testing custom configuration...")
            custom_config = ModelConfig(
                lstm_hidden_size=64,
                transformer_d_model=128,
                xgb_n_estimators=50,
                lgb_n_estimators=50,
                ensemble_weights={'lstm': 0.25, 'transformer': 0.25, 'xgboost': 0.25, 'lightgbm': 0.25}
            )
            
            custom_ensemble = AdvancedMLEnsemble(custom_config)
            print(f"   ✅ Custom configuration applied successfully")
            
            # Test performance monitor
            print("\n3️⃣ Testing performance monitor...")
            monitor = ModelPerformanceMonitor()
            monitor.log_performance('test_model', 0.85, 5.5)
            
            stats = monitor.get_avg_performance('test_model')
            print(f"   ✅ Performance monitor: accuracy={stats['accuracy']:.3f}, time={stats['avg_prediction_time']:.2f}ms")
            
            return True
            
        except Exception as e:
            print(f"❌ Initialization test failed: {str(e)}")
            return False
    
    async def test_feature_engineering(self):
        """Test feature engineering and preparation"""
        print("\n" + "="*80)
        print("🔧 TESTING FEATURE ENGINEERING")
        print("="*80)
        
        try:
            # Generate test data
            self.test_data = self.generate_realistic_market_data(2000)
            
            print("\n📊 Testing feature preparation...")
            
            # Test feature preparation
            features = self.ensemble.prepare_features(self.test_data.copy())
            
            print(f"   ✅ Feature matrix shape: {features.shape}")
            print(f"   ✅ Feature columns: {len(self.ensemble.feature_columns)}")
            print(f"   ✅ Features include: {self.ensemble.feature_columns[:10]}...")
            
            # Test for NaN values
            nan_count = np.isnan(features).sum()
            print(f"   ✅ NaN values handled: {nan_count} remaining")
            
            # Test labels creation
            print("\n🎯 Testing label creation...")
            labels = self.ensemble.create_labels(self.test_data.copy())
            
            unique_labels, counts = np.unique(labels, return_counts=True)
            print(f"   ✅ Label distribution:")
            for label, count in zip(unique_labels, counts):
                label_name = ['SELL', 'HOLD', 'BUY'][int(label)]
                print(f"      {label_name}: {count:,} ({count/len(labels):.1%})")
            
            # Test sequence preparation
            print("\n📈 Testing sequence preparation...")
            seq_features, seq_labels = self.ensemble.prepare_sequences(features, labels, 60)
            
            print(f"   ✅ Sequence shape: {seq_features.shape}")
            print(f"   ✅ Sequence labels: {seq_labels.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Feature engineering test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_model_training(self):
        """Test model training process"""
        print("\n" + "="*80)
        print("🚀 TESTING MODEL TRAINING")
        print("="*80)
        
        try:
            if self.test_data is None:
                self.test_data = self.generate_realistic_market_data(3000)
            
            print("🔄 Starting ensemble training...")
            print("   Note: Using reduced dataset and epochs for testing speed")
            
            # Configure for faster training during testing
            test_config = ModelConfig(
                lstm_hidden_size=32,
                lstm_num_layers=2,
                transformer_d_model=64,
                transformer_nhead=4,
                transformer_num_layers=2,
                xgb_n_estimators=20,
                lgb_n_estimators=20
            )
            
            self.ensemble = AdvancedMLEnsemble(test_config)
            
            start_time = time.time()
            training_results = await self.ensemble.train_models(self.test_data)
            training_time = time.time() - start_time
            
            print(f"\n📊 Training Results:")
            print(f"   Total training time: {training_time:.2f} seconds")
            
            for model_name, results in training_results.items():
                accuracy = results.get('accuracy', 0)
                train_time = results.get('training_time', 0)
                print(f"   {model_name.upper():12} | Accuracy: {accuracy:.4f} | Time: {train_time:.2f}s")
            
            # Verify models are trained
            print(f"\n✅ Ensemble training status: {self.ensemble.is_trained}")
            print(f"✅ Number of models trained: {len(self.ensemble.models)}")
            print(f"✅ Models available: {list(self.ensemble.models.keys())}")
            
            self.performance_results['training'] = training_results
            
            return True
            
        except Exception as e:
            print(f"❌ Model training test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_prediction_speed(self):
        """Test prediction speed and latency requirements"""
        print("\n" + "="*80)
        print("⚡ TESTING PREDICTION SPEED (<10ms TARGET)")
        print("="*80)
        
        try:
            if not self.ensemble.is_trained:
                print("⚠️ Models not trained, skipping speed test")
                return False
            
            # Prepare test features
            features = self.ensemble.prepare_features(self.test_data.copy())
            latest_features = features[-1]  # Most recent data point
            
            print("🔥 Running speed benchmarks...")
            
            # Warm-up predictions (PyTorch compilation, etc.)
            print("   🔄 Warming up models...")
            for _ in range(5):
                await self.ensemble.predict(latest_features)
            
            # Benchmark predictions
            prediction_times = []
            latency_results = {}
            
            print("   ⏱️ Measuring prediction latency...")
            
            for i in range(100):
                start_time = time.perf_counter()
                result = await self.ensemble.predict(latest_features)
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000
                prediction_times.append(latency_ms)
                
                if i == 0:  # Store first result for analysis
                    latency_results = result
            
            # Analyze results
            avg_latency = np.mean(prediction_times)
            p95_latency = np.percentile(prediction_times, 95)
            p99_latency = np.percentile(prediction_times, 99)
            min_latency = np.min(prediction_times)
            max_latency = np.max(prediction_times)
            
            print(f"\n📊 Latency Analysis (100 predictions):")
            print(f"   Average latency: {avg_latency:.2f}ms")
            print(f"   P95 latency: {p95_latency:.2f}ms")
            print(f"   P99 latency: {p99_latency:.2f}ms")
            print(f"   Min latency: {min_latency:.2f}ms")
            print(f"   Max latency: {max_latency:.2f}ms")
            
            # Check if meets requirements
            target_latency = 10.0  # 10ms target
            meets_target = avg_latency < target_latency
            
            print(f"\n🎯 Performance Target Analysis:")
            print(f"   Target: <{target_latency}ms")
            print(f"   Achieved: {avg_latency:.2f}ms")
            print(f"   Status: {'✅ MEETS TARGET' if meets_target else '⚠️ NEEDS OPTIMIZATION'}")
            
            # Model-specific timings
            if 'prediction_times' in latency_results:
                print(f"\n⚙️ Individual Model Timings:")
                for model, timing in latency_results['prediction_times'].items():
                    print(f"   {model.upper():12}: {timing:.2f}ms")
            
            # Sample prediction result
            print(f"\n🔮 Sample Prediction Result:")
            signal_names = ['SELL', 'HOLD', 'BUY']
            signal = latency_results['signal']
            confidence = latency_results['confidence']
            print(f"   Signal: {signal_names[signal]} (confidence: {confidence:.4f})")
            
            self.performance_results['speed'] = {
                'avg_latency_ms': avg_latency,
                'p95_latency_ms': p95_latency,
                'meets_target': meets_target,
                'target_ms': target_latency
            }
            
            return meets_target
            
        except Exception as e:
            print(f"❌ Prediction speed test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_model_accuracy(self):
        """Test model accuracy and performance"""
        print("\n" + "="*80)
        print("🎯 TESTING MODEL ACCURACY & PERFORMANCE")
        print("="*80)
        
        try:
            if not self.ensemble.is_trained:
                print("⚠️ Models not trained, skipping accuracy test")
                return False
            
            # Prepare test data
            features = self.ensemble.prepare_features(self.test_data.copy())
            labels = self.ensemble.create_labels(self.test_data.copy())
            
            # Use recent data for testing
            test_size = 200
            test_features = features[-test_size:]
            test_labels = labels[-test_size:]
            
            print(f"📊 Testing on {test_size} recent data points...")
            
            # Make predictions
            predictions = []
            confidences = []
            
            for i, feature_row in enumerate(test_features):
                result = await self.ensemble.predict(feature_row)
                predictions.append(result['signal'])
                confidences.append(result['confidence'])
                
                if (i + 1) % 50 == 0:
                    print(f"   Processed {i + 1}/{test_size} predictions...")
            
            predictions = np.array(predictions)
            confidences = np.array(confidences)
            
            # Calculate metrics
            accuracy = np.mean(predictions == test_labels)
            avg_confidence = np.mean(confidences)
            
            # Signal distribution
            signal_names = ['SELL', 'HOLD', 'BUY']
            unique_preds, pred_counts = np.unique(predictions, return_counts=True)
            unique_labels, label_counts = np.unique(test_labels, return_counts=True)
            
            print(f"\n📊 Accuracy Results:")
            print(f"   Overall accuracy: {accuracy:.4f} ({accuracy:.1%})")
            print(f"   Average confidence: {avg_confidence:.4f}")
            
            print(f"\n📈 Signal Distribution:")
            print(f"   Predictions vs Actual:")
            for i in range(3):
                pred_count = pred_counts[i] if i in unique_preds else 0
                label_count = label_counts[i] if i in unique_labels else 0
                print(f"   {signal_names[i]:4}: Pred={pred_count:3d} ({pred_count/len(predictions):5.1%}) | "
                      f"Actual={label_count:3d} ({label_count/len(test_labels):5.1%})")
            
            # Performance by confidence level
            high_conf_mask = confidences > 0.6
            if np.sum(high_conf_mask) > 0:
                high_conf_accuracy = np.mean(predictions[high_conf_mask] == test_labels[high_conf_mask])
                print(f"\n🎯 High Confidence Predictions (>60%):")
                print(f"   Count: {np.sum(high_conf_mask)}/{len(predictions)} ({np.sum(high_conf_mask)/len(predictions):.1%})")
                print(f"   Accuracy: {high_conf_accuracy:.4f} ({high_conf_accuracy:.1%})")
            
            self.performance_results['accuracy'] = {
                'overall_accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'high_conf_accuracy': high_conf_accuracy if 'high_conf_accuracy' in locals() else 0
            }
            
            return accuracy > 0.4  # Random would be ~33%, so 40% is reasonable
            
        except Exception as e:
            print(f"❌ Accuracy test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_model_persistence(self):
        """Test model saving and loading"""
        print("\n" + "="*80)
        print("💾 TESTING MODEL PERSISTENCE")
        print("="*80)
        
        try:
            if not self.ensemble.is_trained:
                print("⚠️ Models not trained, skipping persistence test")
                return False
            
            # Test model saving
            print("💾 Testing model saving...")
            save_path = "test_ensemble_model"
            
            self.ensemble.save_models(save_path)
            print("   ✅ Models saved successfully")
            
            # Test model loading
            print("📁 Testing model loading...")
            new_ensemble = AdvancedMLEnsemble()
            new_ensemble.load_models(save_path)
            
            print(f"   ✅ Models loaded successfully")
            print(f"   ✅ Training status: {new_ensemble.is_trained}")
            print(f"   ✅ Models available: {list(new_ensemble.models.keys())}")
            print(f"   ✅ Feature columns: {len(new_ensemble.feature_columns)}")
            
            # Test prediction with loaded model
            print("🔮 Testing prediction with loaded model...")
            features = self.ensemble.prepare_features(self.test_data.copy())
            latest_features = features[-1]
            
            result = await new_ensemble.predict(latest_features)
            print(f"   ✅ Prediction successful: signal={result['signal']}, confidence={result['confidence']:.4f}")
            
            # Cleanup test files
            import glob
            test_files = glob.glob(f"{save_path}*")
            for file in test_files:
                try:
                    os.remove(file)
                except:
                    pass
            
            return True
            
        except Exception as e:
            print(f"❌ Persistence test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    async def run_comprehensive_test(self):
        """Run complete ML ensemble test suite"""
        print("🚀 STARTING ADVANCED ML ENSEMBLE COMPREHENSIVE TEST")
        print("🧠 Testing: LSTM + Transformer + XGBoost + LightGBM Ensemble")
        print("⏰ Test Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        test_results = {}
        
        try:
            # Test 1: Model Initialization
            test_results['initialization'] = await self.test_model_initialization()
            
            # Test 2: Feature Engineering
            test_results['feature_engineering'] = await self.test_feature_engineering()
            
            # Test 3: Model Training
            if test_results['feature_engineering']:
                test_results['model_training'] = await self.test_model_training()
            else:
                test_results['model_training'] = False
            
            # Test 4: Prediction Speed
            if test_results['model_training']:
                test_results['prediction_speed'] = await self.test_prediction_speed()
            else:
                test_results['prediction_speed'] = False
            
            # Test 5: Model Accuracy
            if test_results['model_training']:
                test_results['model_accuracy'] = await self.test_model_accuracy()
            else:
                test_results['model_accuracy'] = False
            
            # Test 6: Model Persistence
            if test_results['model_training']:
                test_results['model_persistence'] = await self.test_model_persistence()
            else:
                test_results['model_persistence'] = False
            
            # Results summary
            print("\n" + "="*80)
            print("📊 ADVANCED ML ENSEMBLE TEST RESULTS")
            print("="*80)
            
            for test_name, result in test_results.items():
                status = "✅ PASSED" if result else "❌ FAILED"
                print(f"{test_name.replace('_', ' ').title():20} | {status}")
            
            overall_success = all(test_results.values())
            
            print(f"\n🏁 Overall Result: {'✅ ALL TESTS PASSED' if overall_success else '⚠️ SOME TESTS FAILED'}")
            
            # Performance summary
            if self.performance_results:
                print(f"\n📊 Performance Summary:")
                
                if 'speed' in self.performance_results:
                    speed = self.performance_results['speed']
                    print(f"   Latency: {speed['avg_latency_ms']:.2f}ms (target: <{speed['target_ms']}ms)")
                
                if 'accuracy' in self.performance_results:
                    acc = self.performance_results['accuracy']
                    print(f"   Accuracy: {acc['overall_accuracy']:.1%}")
                    print(f"   Confidence: {acc['avg_confidence']:.3f}")
                
                if 'training' in self.performance_results:
                    print(f"   Models trained: {len(self.performance_results['training'])}")
            
            print(f"⏰ Test Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            if overall_success:
                print("\n🎉 ADVANCED ML ENSEMBLE IS READY FOR PRODUCTION!")
                print("   🧠 All models trained and operational")
                print("   ⚡ Latency requirements met")
                print("   🎯 Accuracy targets achieved")
                print("   💾 Persistence functionality working")
            
            return overall_success
            
        except Exception as e:
            print(f"\n❌ COMPREHENSIVE TEST FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

async def main():
    """Main test execution"""
    tester = MLEnsembleTestSuite()
    success = await tester.run_comprehensive_test()
    return success

if __name__ == "__main__":
    # Run the comprehensive ML ensemble test
    result = asyncio.run(main())
    exit(0 if result else 1)
