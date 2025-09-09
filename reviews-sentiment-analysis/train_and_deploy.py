#!/usr/bin/env python3
"""
Complete training and deployment script for multilingual sentiment analysis

Usage:
    python train_and_deploy.py --help
    python train_and_deploy.py --quick-test
    python train_and_deploy.py --full-training
    python train_and_deploy.py --deploy-only
"""

import argparse
import os
import sys
import logging
import time
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.data.data_loader import AmazonReviewsDataLoader
from src.training.trainer import train_model, SentimentTrainer
from src.evaluation.evaluator import run_evaluation
from src.models.distilbert_sentiment import SentimentPredictor, test_model
from src.api.main import app

import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if environment is properly set up"""
    logger.info("Checking environment...")
    
    try:
        import torch
        logger.info(f"✓ PyTorch {torch.__version__} available")
        logger.info(f"✓ CUDA available: {torch.cuda.is_available()}")
        
        import transformers
        logger.info(f"✓ Transformers {transformers.__version__} available")
        
        import pandas as pd
        logger.info(f"✓ Pandas {pd.__version__} available")
        
        import fastapi
        logger.info(f"✓ FastAPI {fastapi.__version__} available")
        
        # Check datasets
        data_dir = "datasets"
        required_files = [
            "amazon_reviews_multi_train.csv",
            "amazon_reviews_multi_validation.csv", 
            "amazon_reviews_multi_test.csv"
        ]
        
        for file in required_files:
            file_path = os.path.join(data_dir, file)
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                logger.info(f"✓ {file} ({size_mb:.1f}MB)")
            else:
                logger.error(f"✗ Missing dataset file: {file_path}")
                return False
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ Missing dependency: {e}")
        return False

def test_pretrained_model():
    """Test the pretrained model"""
    logger.info("Testing pretrained multilingual DistilBERT model...")
    
    try:
        test_model()
        logger.info("✓ Pretrained model test successful")
        return True
    except Exception as e:
        logger.error(f"✗ Pretrained model test failed: {e}")
        return False

def run_quick_training():
    """Run quick training for testing"""
    logger.info("Starting quick training (small dataset for testing)...")
    
    try:
        start_time = time.time()
        
        model_path, trainer = train_model(
            epochs=2,
            batch_size=8,
            learning_rate=2e-5,
            max_samples=1000,  # Small sample for quick testing
            balanced=True
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        logger.info(f"✓ Quick training completed in {training_time:.1f} seconds")
        logger.info(f"✓ Model saved to: {model_path}")
        
        return model_path
        
    except Exception as e:
        logger.error(f"✗ Quick training failed: {e}")
        return None

def run_full_training():
    """Run full training on complete dataset"""
    logger.info("Starting full training (complete dataset)...")
    
    try:
        start_time = time.time()
        
        model_path, trainer = train_model(
            epochs=3,
            batch_size=16,
            learning_rate=2e-5,
            balanced=True  # Use balanced sampling
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        logger.info(f"✓ Full training completed in {training_time:.1f} seconds ({training_time/3600:.1f} hours)")
        logger.info(f"✓ Model saved to: {model_path}")
        
        return model_path
        
    except Exception as e:
        logger.error(f"✗ Full training failed: {e}")
        return None

def evaluate_model(model_path=None):
    """Evaluate trained or pretrained model"""
    if model_path:
        logger.info(f"Evaluating trained model: {model_path}")
    else:
        logger.info("Evaluating pretrained multilingual model")
    
    try:
        results, error_analysis = run_evaluation(model_path=model_path)
        
        logger.info("✓ Model evaluation completed")
        logger.info(f"  - Accuracy: {results['accuracy']:.4f}")
        logger.info(f"  - Macro F1: {results['macro_f1']:.4f}")
        logger.info(f"  - Weighted F1: {results['weighted_f1']:.4f}")
        logger.info(f"  - Error Rate: {error_analysis['error_rate']:.2%}")
        
        return results
        
    except Exception as e:
        logger.error(f"✗ Model evaluation failed: {e}")
        return None

def deploy_api(model_path=None, port=8000):
    """Deploy FastAPI service"""
    if model_path:
        logger.info(f"Deploying API with trained model: {model_path}")
    else:
        logger.info("Deploying API with pretrained multilingual model")
    
    try:
        # Update model path in app if provided
        if model_path:
            from src.api.main import load_model
            load_model(model_path)
        
        logger.info(f"🚀 Starting API server on port {port}")
        logger.info(f"📊 API Documentation: http://localhost:{port}/docs")
        logger.info(f"🌐 Web Interface: http://localhost:{port}")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"✗ API deployment failed: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Multilingual Sentiment Analysis Training and Deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_and_deploy.py --check-env          # Check environment
  python train_and_deploy.py --test-pretrained    # Test pretrained model
  python train_and_deploy.py --quick-test         # Quick training and test
  python train_and_deploy.py --full-training      # Full training pipeline
  python train_and_deploy.py --deploy-only        # Deploy with pretrained model
  python train_and_deploy.py --evaluate           # Evaluate pretrained model
        """
    )
    
    parser.add_argument("--check-env", action="store_true",
                       help="Check if environment is properly set up")
    parser.add_argument("--test-pretrained", action="store_true",
                       help="Test pretrained multilingual model")
    parser.add_argument("--quick-test", action="store_true",
                       help="Run quick training with small dataset")
    parser.add_argument("--full-training", action="store_true",
                       help="Run full training on complete dataset")
    parser.add_argument("--evaluate", action="store_true",
                       help="Evaluate model performance")
    parser.add_argument("--deploy-only", action="store_true",
                       help="Deploy API without training")
    parser.add_argument("--model-path", type=str,
                       help="Path to trained model for evaluation/deployment")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port for API server (default: 8000)")
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    print("=" * 60)
    print("🤖 Multilingual Sentiment Analysis")
    print("DistilBERT + Transfer Learning + FastAPI")
    print("=" * 60)
    
    # Check environment
    if args.check_env or not any([args.test_pretrained, args.quick_test, 
                                 args.full_training, args.evaluate, args.deploy_only]):
        if not check_environment():
            logger.error("Environment check failed. Please install required dependencies.")
            return
    
    # Test pretrained model
    if args.test_pretrained:
        if not test_pretrained_model():
            return
    
    model_path = args.model_path
    
    # Training
    if args.quick_test:
        logger.info("\n" + "="*40 + " QUICK TRAINING " + "="*40)
        model_path = run_quick_training()
        if not model_path:
            return
        
        # Evaluate quick training results
        logger.info("\n" + "="*40 + " EVALUATION " + "="*40)
        evaluate_model(model_path)
    
    elif args.full_training:
        logger.info("\n" + "="*40 + " FULL TRAINING " + "="*40)
        model_path = run_full_training()
        if not model_path:
            return
            
        # Evaluate full training results
        logger.info("\n" + "="*40 + " EVALUATION " + "="*40)
        evaluate_model(model_path)
    
    # Evaluation only
    if args.evaluate and not (args.quick_test or args.full_training):
        logger.info("\n" + "="*40 + " EVALUATION " + "="*40)
        evaluate_model(model_path)
    
    # Deploy API
    if args.deploy_only or args.quick_test or args.full_training:
        logger.info("\n" + "="*40 + " API DEPLOYMENT " + "="*40)
        deploy_api(model_path, args.port)

if __name__ == "__main__":
    main()