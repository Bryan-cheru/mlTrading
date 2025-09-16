"""
Deployment Validation Script
Automated testing and validation for Phase 1 system deployment
"""

import sys
import os
import time
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Tuple

# Configure logging for validation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - VALIDATOR - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeploymentValidator:
    """Automated validation for system deployment"""
    
    def __init__(self):
        self.validation_results = {}
        self.start_time = datetime.now()
        
    def print_header(self):
        """Print validation header"""
        print("=" * 80)
        print("🚀 PHASE 1 SYSTEM - DEPLOYMENT VALIDATION")
        print("=" * 80)
        print(f"Start Time: {self.start_time}")
        print()

    def validate_environment(self) -> bool:
        """Validate Python environment and dependencies"""
        logger.info("🔍 Validating Python environment...")
        
        try:
            # Check Python version
            python_version = sys.version_info
            if python_version.major >= 3 and python_version.minor >= 8:
                logger.info(f"✅ Python version: {python_version.major}.{python_version.minor}")
            else:
                logger.error(f"❌ Python version too old: {python_version.major}.{python_version.minor}")
                return False
            
            # Check critical imports
            critical_modules = [
                'numpy', 'pandas', 'xgboost', 'asyncio', 
                'threading', 'socket', 'json'
            ]
            
            for module in critical_modules:
                try:
                    __import__(module)
                    logger.info(f"✅ Module {module}: Available")
                except ImportError:
                    logger.error(f"❌ Module {module}: Missing")
                    return False
            
            self.validation_results['environment'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Environment validation failed: {e}")
            self.validation_results['environment'] = False
            return False

    def validate_project_structure(self) -> bool:
        """Validate project directory structure"""
        logger.info("🔍 Validating project structure...")
        
        try:
            required_files = [
                'simplified_advanced_system.py',
                'test_production_system.py',
                'data-pipeline/ingestion/ninjatrader_connector.py',
                'requirements.txt'
            ]
            
            missing_files = []
            for file_path in required_files:
                if os.path.exists(file_path):
                    logger.info(f"✅ File exists: {file_path}")
                else:
                    logger.error(f"❌ File missing: {file_path}")
                    missing_files.append(file_path)
            
            if missing_files:
                logger.error(f"❌ Missing files: {missing_files}")
                self.validation_results['project_structure'] = False
                return False
            
            self.validation_results['project_structure'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Project structure validation failed: {e}")
            self.validation_results['project_structure'] = False
            return False

    def validate_system_components(self) -> bool:
        """Validate system components can be imported"""
        logger.info("🔍 Validating system components...")
        
        try:
            # Add path for imports
            sys.path.append(os.path.join(os.path.dirname(__file__), 'data-pipeline', 'ingestion'))
            
            # Test core imports
            from simplified_advanced_system import (
                AdvancedTradingSystem, 
                EnhancedFeatureEngine,
                SimpleMLModel,
                SimplePortfolioManager
            )
            
            logger.info("✅ Core system components: Importable")
            
            # Test component initialization
            feature_engine = EnhancedFeatureEngine()
            ml_model = SimpleMLModel()
            portfolio = SimplePortfolioManager()
            
            logger.info("✅ Component initialization: Success")
            
            self.validation_results['system_components'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ System components validation failed: {e}")
            self.validation_results['system_components'] = False
            return False

    def validate_ninjatrader_integration(self) -> bool:
        """Test NinjaTrader connector (without requiring connection)"""
        logger.info("🔍 Validating NinjaTrader integration...")
        
        try:
            from ninjatrader_connector import NinjaTraderConnector, MarketData
            from datetime import datetime
            
            # Test connector initialization
            connector = NinjaTraderConnector()
            logger.info("✅ NinjaTrader connector: Initialized")
            
            # Test MarketData structure
            test_data = MarketData(
                instrument='ES 12-24',
                timestamp=datetime.now(),
                bid=4500.0,
                ask=4500.25,
                last=4500.125,
                volume=100,
                bid_size=10,
                ask_size=10
            )
            logger.info("✅ MarketData structure: Valid")
            
            self.validation_results['ninjatrader_integration'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ NinjaTrader integration validation failed: {e}")
            self.validation_results['ninjatrader_integration'] = False
            return False

    def run_production_tests(self) -> bool:
        """Run the production test suite"""
        logger.info("🔍 Running production test suite...")
        
        try:
            # Import and run production tests
            import subprocess
            import sys
            
            # Run the production test script
            result = subprocess.run(
                [sys.executable, 'test_production_safe.py'],
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            
            if result.returncode == 0:
                logger.info("✅ Production tests: ALL PASSED")
                # Check for success indicators in output
                if "100.0%" in result.stdout and "ALL TESTS PASSED" in result.stdout:
                    self.validation_results['production_tests'] = True
                    return True
            
            logger.error(f"❌ Production tests failed. Return code: {result.returncode}")
            if result.stderr:
                logger.error(f"Error output: {result.stderr}")
            
            self.validation_results['production_tests'] = False
            return False
            
        except subprocess.TimeoutExpired:
            logger.error("❌ Production tests timed out")
            self.validation_results['production_tests'] = False
            return False
        except Exception as e:
            logger.error(f"❌ Production tests validation failed: {e}")
            self.validation_results['production_tests'] = False
            return False

    def validate_configuration(self) -> bool:
        """Validate system configuration"""
        logger.info("🔍 Validating system configuration...")
        
        try:
            # Check if config directory exists
            config_dir = 'config'
            if not os.path.exists(config_dir):
                logger.warning("⚠️ Config directory missing - using defaults")
            
            # Test system initialization
            from simplified_advanced_system import AdvancedTradingSystem
            system = AdvancedTradingSystem()
            
            # Validate configuration values
            if hasattr(system, 'instruments') and system.instruments:
                logger.info(f"✅ Instruments configured: {system.instruments}")
            else:
                logger.error("❌ No instruments configured")
                return False
            
            if hasattr(system, 'min_confidence') and 0 < system.min_confidence < 1:
                logger.info(f"✅ Confidence threshold: {system.min_confidence}")
            else:
                logger.error("❌ Invalid confidence threshold")
                return False
            
            self.validation_results['configuration'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            self.validation_results['configuration'] = False
            return False

    def check_ninjatrader_connection(self) -> Tuple[bool, str]:
        """Check if NinjaTrader is running and accepting connections"""
        logger.info("🔍 Checking NinjaTrader 8 availability...")
        
        try:
            import socket
            
            # Test connection to NinjaTrader port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('127.0.0.1', 36973))
            sock.close()
            
            if result == 0:
                logger.info("✅ NinjaTrader 8: Running and accepting connections")
                return True, "NinjaTrader 8 is running with ATI enabled"
            else:
                logger.warning("⚠️ NinjaTrader 8: Not accessible on port 36973")
                return False, "NinjaTrader 8 not running or ATI not enabled"
                
        except Exception as e:
            logger.warning(f"⚠️ NinjaTrader connection check failed: {e}")
            return False, f"Connection check error: {e}"

    def print_deployment_summary(self):
        """Print final deployment validation summary"""
        print("\n" + "=" * 80)
        print("📊 DEPLOYMENT VALIDATION SUMMARY")
        print("=" * 80)
        
        total_checks = len(self.validation_results)
        passed_checks = sum(1 for result in self.validation_results.values() if result)
        
        for check_name, result in self.validation_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            formatted_name = check_name.replace('_', ' ').title()
            print(f"{formatted_name}: {status}")
        
        print("-" * 80)
        print(f"Total Checks: {total_checks}")
        print(f"Passed: {passed_checks}")
        print(f"Failed: {total_checks - passed_checks}")
        print(f"Success Rate: {(passed_checks/total_checks)*100:.1f}%")
        
        # Check NinjaTrader separately (optional)
        nt_available, nt_message = self.check_ninjatrader_connection()
        print(f"\nNinjaTrader 8 Status: {nt_message}")
        
        # Final recommendation
        if passed_checks == total_checks:
            print("\n🎉 SYSTEM READY FOR DEPLOYMENT!")
            print("✅ All validation checks passed")
            if nt_available:
                print("✅ NinjaTrader 8 is ready for live trading")
                print("\n🚀 DEPLOYMENT COMMANDS:")
                print("python simplified_advanced_system.py")
            else:
                print("⚠️ Start NinjaTrader 8 with ATI enabled for live trading")
                print("\n🚀 DEMO MODE AVAILABLE:")
                print("python ninjatrader_demo.py")
        else:
            print("\n⚠️ DEPLOYMENT NOT RECOMMENDED")
            print("❌ Some validation checks failed")
            print("🔧 Fix the failed checks before deploying")
        
        print("=" * 80)

def main():
    """Main validation routine"""
    validator = DeploymentValidator()
    validator.print_header()
    
    # Run all validation checks
    validation_checks = [
        validator.validate_environment,
        validator.validate_project_structure,
        validator.validate_system_components,
        validator.validate_ninjatrader_integration,
        validator.validate_configuration,
        validator.run_production_tests
    ]
    
    print("Running validation checks...\n")
    
    for check in validation_checks:
        success = check()
        time.sleep(0.5)  # Brief pause between checks
        if not success:
            print(f"\n⚠️ Validation check failed: {check.__name__}")
    
    # Print final summary
    validator.print_deployment_summary()

if __name__ == "__main__":
    main()
