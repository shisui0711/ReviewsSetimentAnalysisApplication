"""
Usage examples for the Multilingual Sentiment Analysis system

This file demonstrates various ways to use the sentiment analysis system:
1. Basic sentiment prediction
2. Batch processing
3. Multilingual examples
4. API client examples
5. Custom training examples
"""

import os
import sys
import pandas as pd
import requests
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.models.distilbert_sentiment import SentimentPredictor
from src.training.trainer import train_model
from src.evaluation.evaluator import run_evaluation

def example_basic_prediction():
    """Example 1: Basic sentiment prediction"""
    print("=" * 60)
    print("Example 1: Basic Sentiment Prediction")
    print("=" * 60)
    
    # Initialize predictor
    predictor = SentimentPredictor()
    
    # Test texts
    texts = [
        "I absolutely love this product! It's amazing and works perfectly.",
        "The quality is okay, nothing special but does the job.",
        "Terrible product, completely waste of money. Very disappointed.",
        "Great customer service and fast shipping. Highly recommended!",
        "Average product, could be better but not bad either."
    ]
    
    print("Analyzing sample texts...\n")
    
    for i, text in enumerate(texts, 1):
        sentiment, probs = predictor.predict_single(text, return_probabilities=True)
        confidence = max(probs)
        
        print(f"Text {i}: {text[:50]}...")
        print(f"Sentiment: {sentiment.upper()} (confidence: {confidence:.3f})")
        print(f"Probabilities: neg={probs[0]:.3f}, neu={probs[1]:.3f}, pos={probs[2]:.3f}")
        print("-" * 60)

def example_batch_processing():
    """Example 2: Batch processing multiple texts"""
    print("\n" + "=" * 60)
    print("Example 2: Batch Processing")
    print("=" * 60)
    
    # Initialize predictor
    predictor = SentimentPredictor()
    
    # Create sample dataset
    sample_reviews = [
        "Outstanding product quality! Exceeded my expectations completely.",
        "Good value for money, decent quality overall.",
        "Poor build quality, broke after just one week of use.",
        "Excellent customer support, very helpful and responsive.",
        "Average product, nothing extraordinary about it.",
        "Fantastic design and very user-friendly interface.",
        "Overpriced for what you get, not worth the money.",
        "Solid product, reliable and durable construction.",
        "Disappointing performance, did not meet my needs.",
        "Perfect for my requirements, works exactly as described."
    ]
    
    print(f"Processing batch of {len(sample_reviews)} reviews...\n")
    
    # Batch prediction
    start_time = time.time()
    sentiments, probabilities = predictor.predict(sample_reviews, return_probabilities=True)
    end_time = time.time()
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'review': sample_reviews,
        'sentiment': sentiments,
        'confidence': [max(probs) for probs in probabilities],
        'neg_prob': [probs[0] for probs in probabilities],
        'neu_prob': [probs[1] for probs in probabilities],
        'pos_prob': [probs[2] for probs in probabilities]
    })
    
    print("Batch Processing Results:")
    print(results_df.to_string(index=False))
    print(f"\nProcessing time: {(end_time - start_time)*1000:.1f}ms")
    print(f"Average time per text: {((end_time - start_time)/len(sample_reviews))*1000:.1f}ms")
    
    # Save results
    output_file = "batch_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")

def example_multilingual():
    """Example 3: Multilingual sentiment analysis"""
    print("\n" + "=" * 60)
    print("Example 3: Multilingual Analysis")
    print("=" * 60)
    
    # Initialize predictor
    predictor = SentimentPredictor()
    
    # Multilingual examples
    multilingual_texts = [
        ("English", "This product is fantastic! Highly recommend it."),
        ("Spanish", "Este producto es fantástico! Lo recomiendo mucho."),
        ("French", "Ce produit est fantastique! Je le recommande vivement."),
        ("German", "Dieses Produkt ist fantastisch! Ich empfehle es sehr."),
        ("Portuguese", "Este produto é fantástico! Recomendo muito."),
        ("Italian", "Questo prodotto è fantastico! Lo consiglio vivamente."),
        ("English", "Terrible quality, complete waste of money."),
        ("Spanish", "Terrible calidad, completa pérdida de dinero."),
        ("French", "Terrible qualité, perte d'argent complète."),
        ("German", "Schreckliche Qualität, komplette Geldverschwendung."),
        ("Portuguese", "Qualidade terrível, completo desperdício de dinheiro."),
        ("Italian", "Qualità terribile, completo spreco di denaro.")
    ]
    
    print("Analyzing multilingual reviews...\n")
    
    for language, text in multilingual_texts:
        sentiment, probs = predictor.predict_single(text, return_probabilities=True)
        confidence = max(probs)
        
        print(f"Language: {language}")
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment.upper()} (confidence: {confidence:.3f})")
        print("-" * 60)

def example_api_client():
    """Example 4: API client usage"""
    print("\n" + "=" * 60)
    print("Example 4: API Client Usage")
    print("=" * 60)
    
    api_base_url = "http://localhost:8000"
    
    print("Note: This example requires the API server to be running.")
    print("Start the server with: python src/api/main.py")
    print("Or: python train_and_deploy.py --deploy-only\n")
    
    # Test API endpoints
    test_cases = [
        {
            "endpoint": "/health",
            "method": "GET",
            "description": "Health check"
        },
        {
            "endpoint": "/analyze", 
            "method": "POST",
            "data": {"text": "This product is amazing!"},
            "description": "Single text analysis"
        },
        {
            "endpoint": "/analyze/batch",
            "method": "POST", 
            "data": {"texts": ["Great product!", "Okay quality", "Poor service"]},
            "description": "Batch text analysis"
        },
        {
            "endpoint": "/model/info",
            "method": "GET",
            "description": "Model information"
        }
    ]
    
    for test_case in test_cases:
        print(f"Testing: {test_case['description']}")
        print(f"Endpoint: {test_case['method']} {test_case['endpoint']}")
        
        try:
            if test_case['method'] == 'GET':
                response = requests.get(f"{api_base_url}{test_case['endpoint']}", timeout=5)
            else:
                response = requests.post(
                    f"{api_base_url}{test_case['endpoint']}", 
                    json=test_case['data'],
                    timeout=5
                )
            
            if response.status_code == 200:
                print("✓ Success")
                if 'analyze' in test_case['endpoint']:
                    result = response.json()
                    if 'sentiment' in result:
                        print(f"  Sentiment: {result['sentiment']}")
                        print(f"  Confidence: {result['confidence']:.3f}")
                    elif 'results' in result:
                        print(f"  Processed {len(result['results'])} texts")
            else:
                print(f"✗ Error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("✗ Connection failed (API server not running)")
        except Exception as e:
            print(f"✗ Error: {e}")
        
        print("-" * 60)

def example_custom_training():
    """Example 5: Custom training pipeline"""
    print("\n" + "=" * 60)
    print("Example 5: Custom Training Pipeline")
    print("=" * 60)
    
    print("This example demonstrates custom training with specific parameters.")
    print("Warning: Training can take significant time depending on dataset size.\n")
    
    # Training configuration
    training_configs = [
        {
            "name": "Quick Test Training",
            "params": {
                "epochs": 1,
                "batch_size": 4,
                "learning_rate": 2e-5,
                "max_samples": 100,
                "balanced": True
            }
        },
        {
            "name": "Development Training",
            "params": {
                "epochs": 2,
                "batch_size": 8,
                "learning_rate": 2e-5,
                "max_samples": 1000,
                "balanced": True
            }
        }
    ]
    
    for config in training_configs:
        print(f"Configuration: {config['name']}")
        print(f"Parameters: {config['params']}")
        
        # Ask user confirmation
        response = input("Run this training? (y/n): ").lower()
        
        if response == 'y':
            print("Starting training...")
            start_time = time.time()
            
            try:
                model_path, trainer = train_model(**config['params'])
                end_time = time.time()
                
                print(f"✓ Training completed in {end_time - start_time:.1f}s")
                print(f"✓ Model saved to: {model_path}")
                
                # Quick evaluation
                print("Running quick evaluation...")
                results, _ = run_evaluation(model_path=model_path)
                print(f"✓ Accuracy: {results['accuracy']:.4f}")
                
            except Exception as e:
                print(f"✗ Training failed: {e}")
        else:
            print("Skipped.")
            
        print("-" * 60)

def example_error_analysis():
    """Example 6: Model error analysis"""
    print("\n" + "=" * 60)
    print("Example 6: Model Error Analysis")
    print("=" * 60)
    
    # This would typically be done with a test dataset
    print("Running comprehensive model evaluation...")
    
    try:
        results, error_analysis = run_evaluation()
        
        print("\nModel Performance Summary:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Macro F1: {results['macro_f1']:.4f}")
        print(f"Weighted F1: {results['weighted_f1']:.4f}")
        
        print("\nError Analysis:")
        print(f"Total Errors: {error_analysis['total_errors']}")
        print(f"Error Rate: {error_analysis['error_rate']:.2%}")
        print(f"High Confidence Errors: {error_analysis['high_confidence_errors']}")
        
        print("\nCommon Confusion Patterns:")
        for pattern, count in error_analysis['common_confusions'].items():
            print(f"  {pattern}: {count}")
            
        print("\nSample Errors (for manual inspection):")
        for i, error in enumerate(error_analysis['sample_errors'][:3]):
            print(f"{i+1}. Text: {error['text'][:80]}...")
            print(f"   True: {error['true_sentiment']} | Predicted: {error['predicted_sentiment']}")
            print(f"   Confidence: {error['confidence']:.3f}")
            
    except Exception as e:
        print(f"Error analysis failed: {e}")

def main():
    """Run all examples"""
    print("🤖 Multilingual Sentiment Analysis - Usage Examples")
    print("=" * 80)
    
    examples = [
        ("Basic Prediction", example_basic_prediction),
        ("Batch Processing", example_batch_processing), 
        ("Multilingual Analysis", example_multilingual),
        ("API Client", example_api_client),
        ("Custom Training", example_custom_training),
        ("Error Analysis", example_error_analysis)
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")
    
    print("\nSelect examples to run:")
    print("- Enter numbers separated by commas (e.g., 1,2,3)")
    print("- Enter 'all' to run all examples")
    print("- Enter 'quit' to exit")
    
    while True:
        choice = input("\nYour choice: ").strip().lower()
        
        if choice == 'quit':
            break
        elif choice == 'all':
            for name, func in examples:
                print(f"\n{'='*20} {name} {'='*20}")
                try:
                    func()
                except KeyboardInterrupt:
                    print("\nSkipped by user.")
                except Exception as e:
                    print(f"Error in {name}: {e}")
            break
        else:
            try:
                selected = [int(x.strip()) for x in choice.split(',')]
                for num in selected:
                    if 1 <= num <= len(examples):
                        name, func = examples[num - 1]
                        print(f"\n{'='*20} {name} {'='*20}")
                        try:
                            func()
                        except KeyboardInterrupt:
                            print("\nSkipped by user.")
                        except Exception as e:
                            print(f"Error in {name}: {e}")
                    else:
                        print(f"Invalid choice: {num}")
                break
            except ValueError:
                print("Invalid input. Please enter numbers or 'all' or 'quit'.")

if __name__ == "__main__":
    main()