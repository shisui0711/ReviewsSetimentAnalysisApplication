import torch
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm

from ..models.distilbert_sentiment import SentimentPredictor
from ..data.data_loader import AmazonReviewsDataLoader

logger = logging.getLogger(__name__)

class SentimentEvaluator:
    """
    Comprehensive evaluation for sentiment analysis models
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.predictor = SentimentPredictor(model_path) if model_path else SentimentPredictor()
        self.label_names = ["negative", "neutral", "positive"]
        
    def evaluate_dataset(
        self, 
        test_df: pd.DataFrame, 
        batch_size: int = 32
    ) -> Dict:
        """
        Evaluate model performance on test dataset
        
        Args:
            test_df: Test dataframe with 'text' and 'sentiment' columns
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Evaluating model on {len(test_df)} samples")
        
        # Prepare data
        texts = test_df['text'].tolist()
        true_labels = test_df['sentiment'].tolist()
        
        # Get predictions in batches
        predicted_labels = []
        predicted_probs = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating"):
            batch_texts = texts[i:i + batch_size]
            
            # Get predictions
            labels, probs = self.predictor.predict(batch_texts, return_probabilities=True)
            
            # Convert labels to integers
            label_to_int = {"negative": 0, "neutral": 1, "positive": 2}
            batch_labels = [label_to_int[label] for label in labels]
            
            predicted_labels.extend(batch_labels)
            predicted_probs.extend(probs)
        
        # Calculate metrics
        results = self._calculate_metrics(true_labels, predicted_labels, predicted_probs)
        
        # Add sample predictions for analysis
        results['sample_predictions'] = self._get_sample_predictions(
            texts[:10], true_labels[:10], predicted_labels[:10], predicted_probs[:10]
        )
        
        return results
    
    def _calculate_metrics(
        self, 
        true_labels: List[int], 
        predicted_labels: List[int], 
        predicted_probs: List[np.ndarray]
    ) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        
        # Basic metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predicted_labels, average=None
        )
        
        # Macro and weighted averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average='macro'
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average='weighted'
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # Classification report
        report = classification_report(
            true_labels, predicted_labels,
            target_names=self.label_names,
            output_dict=True
        )
        
        # Confidence statistics
        confidences = [max(probs) for probs in predicted_probs]
        
        results = {
            'accuracy': float(accuracy),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1),
            'weighted_precision': float(weighted_precision),
            'weighted_recall': float(weighted_recall),
            'weighted_f1': float(weighted_f1),
            'per_class_metrics': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'f1': f1.tolist(),
                'support': support.tolist()
            },
            'confusion_matrix': cm.tolist(),
            'classification_report': self._convert_report_to_json_serializable(report),
            'confidence_stats': {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences))
            }
        }
        
        return results
    
    def _convert_report_to_json_serializable(self, report: Dict) -> Dict:
        """Convert classification report to JSON serializable format"""
        serializable_report = {}
        for key, value in report.items():
            if isinstance(value, dict):
                serializable_report[key] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in value.items()
                }
            else:
                serializable_report[key] = float(value) if isinstance(value, (np.floating, np.integer)) else value
        return serializable_report
    
    def _get_sample_predictions(
        self, 
        texts: List[str], 
        true_labels: List[int], 
        predicted_labels: List[int], 
        predicted_probs: List[np.ndarray]
    ) -> List[Dict]:
        """Get sample predictions for qualitative analysis"""
        
        samples = []
        for text, true_label, pred_label, probs in zip(texts, true_labels, predicted_labels, predicted_probs):
            samples.append({
                'text': text[:100] + "..." if len(text) > 100 else text,
                'true_sentiment': self.label_names[true_label],
                'predicted_sentiment': self.label_names[pred_label],
                'confidence': float(max(probs)),
                'correct': true_label == pred_label,
                'probabilities': {
                    'negative': float(probs[0]),
                    'neutral': float(probs[1]),
                    'positive': float(probs[2])
                }
            })
        
        return samples
    
    def plot_confusion_matrix(self, confusion_matrix: List[List[int]], save_path: Optional[str] = None):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.label_names,
            yticklabels=self.label_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_class_distribution(self, true_labels: List[int], predicted_labels: List[int], save_path: Optional[str] = None):
        """Plot class distribution comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # True labels distribution
        true_counts = [true_labels.count(i) for i in range(3)]
        ax1.bar(self.label_names, true_counts, alpha=0.7, color='skyblue')
        ax1.set_title('True Label Distribution')
        ax1.set_ylabel('Count')
        
        # Predicted labels distribution
        pred_counts = [predicted_labels.count(i) for i in range(3)]
        ax2.bar(self.label_names, pred_counts, alpha=0.7, color='lightcoral')
        ax2.set_title('Predicted Label Distribution')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def analyze_errors(self, test_df: pd.DataFrame, results: Dict) -> Dict:
        """Analyze model errors for insights"""
        
        # Get misclassified samples
        texts = test_df['text'].tolist()
        true_labels = test_df['sentiment'].tolist()
        
        # Re-predict to get detailed info
        predicted_labels = []
        predicted_probs = []
        
        for text in tqdm(texts, desc="Re-analyzing for errors"):
            label, probs = self.predictor.predict_single(text, return_probabilities=True)
            label_to_int = {"negative": 0, "neutral": 1, "positive": 2}
            predicted_labels.append(label_to_int[label])
            predicted_probs.append(probs)
        
        # Find errors
        errors = []
        for i, (text, true_label, pred_label, probs) in enumerate(
            zip(texts, true_labels, predicted_labels, predicted_probs)
        ):
            if true_label != pred_label:
                errors.append({
                    'index': i,
                    'text': text,
                    'true_sentiment': self.label_names[true_label],
                    'predicted_sentiment': self.label_names[pred_label],
                    'confidence': float(max(probs)),
                    'probabilities': {
                        'negative': float(probs[0]),
                        'neutral': float(probs[1]),
                        'positive': float(probs[2])
                    }
                })
        
        # Analyze error patterns
        error_patterns = {
            'total_errors': len(errors),
            'error_rate': float(len(errors) / len(texts)),
            'high_confidence_errors': len([e for e in errors if e['confidence'] > 0.8]),
            'low_confidence_errors': len([e for e in errors if e['confidence'] < 0.6]),
            'common_confusions': self._analyze_confusion_patterns(true_labels, predicted_labels),
            'sample_errors': errors[:10]  # Top 10 errors for manual inspection
        }
        
        return error_patterns
    
    def _analyze_confusion_patterns(self, true_labels: List[int], predicted_labels: List[int]) -> Dict:
        """Analyze common confusion patterns"""
        
        confusions = {}
        for true_label, pred_label in zip(true_labels, predicted_labels):
            if true_label != pred_label:
                key = f"{self.label_names[true_label]} -> {self.label_names[pred_label]}"
                confusions[key] = confusions.get(key, 0) + 1
        
        # Sort by frequency
        sorted_confusions = dict(sorted(confusions.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_confusions
    
    def generate_evaluation_report(self, results: Dict, save_path: Optional[str] = None) -> str:
        """Generate a comprehensive evaluation report"""
        
        report = f"""
# Multilingual Sentiment Analysis Model Evaluation Report

## Overall Performance
- **Accuracy**: {results['accuracy']:.4f}
- **Macro F1 Score**: {results['macro_f1']:.4f}
- **Weighted F1 Score**: {results['weighted_f1']:.4f}

## Per-Class Performance
"""
        
        for i, class_name in enumerate(self.label_names):
            report += f"""
### {class_name.capitalize()} Class
- Precision: {results['per_class_metrics']['precision'][i]:.4f}
- Recall: {results['per_class_metrics']['recall'][i]:.4f}
- F1 Score: {results['per_class_metrics']['f1'][i]:.4f}
- Support: {results['per_class_metrics']['support'][i]}
"""
        
        report += f"""
## Confidence Statistics
- Mean Confidence: {results['confidence_stats']['mean']:.4f}
- Confidence Std: {results['confidence_stats']['std']:.4f}
- Min Confidence: {results['confidence_stats']['min']:.4f}
- Max Confidence: {results['confidence_stats']['max']:.4f}

## Sample Predictions
"""
        
        for sample in results['sample_predictions']:
            status = "✓" if sample['correct'] else "✗"
            report += f"""
{status} **Text**: {sample['text']}
   **True**: {sample['true_sentiment']} | **Predicted**: {sample['predicted_sentiment']} 
   **Confidence**: {sample['confidence']:.3f}
"""
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report

def run_evaluation(
    model_path: Optional[str] = None,
    data_dir: str = "datasets",
    output_dir: str = "evaluation_results"
):
    """
    Run comprehensive evaluation on test dataset
    
    Args:
        model_path: Path to trained model (None for pretrained)
        data_dir: Directory containing datasets
        output_dir: Directory to save evaluation results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data
    data_loader = AmazonReviewsDataLoader(data_dir)
    all_splits = data_loader.load_all_splits(balanced=False)
    
    if 'test' not in all_splits:
        logger.warning("No test split found, using validation split")
        test_df = all_splits.get('validation')
    else:
        test_df = all_splits['test']
    
    if test_df is None or len(test_df) == 0:
        raise ValueError("No test data available")
    
    logger.info(f"Running evaluation on {len(test_df)} test samples")
    
    # Initialize evaluator
    evaluator = SentimentEvaluator(model_path)
    
    # Run evaluation
    results = evaluator.evaluate_dataset(test_df)
    
    # Plot results
    evaluator.plot_confusion_matrix(
        results['confusion_matrix'],
        save_path=os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    # Generate report
    report = evaluator.generate_evaluation_report(
        results,
        save_path=os.path.join(output_dir, 'evaluation_report.md')
    )
    
    # Analyze errors
    error_analysis = evaluator.analyze_errors(test_df, results)
    
    # Save detailed results
    import json
    with open(os.path.join(output_dir, 'detailed_results.json'), 'w') as f:
        json.dump({
            'metrics': results,
            'error_analysis': error_analysis
        }, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print(f"Weighted F1: {results['weighted_f1']:.4f}")
    print(f"Mean Confidence: {results['confidence_stats']['mean']:.4f}")
    print(f"Total Errors: {error_analysis['total_errors']}")
    print(f"Error Rate: {error_analysis['error_rate']:.2%}")
    print("="*60)
    
    return results, error_analysis

if __name__ == "__main__":
    # Run evaluation
    results, error_analysis = run_evaluation()
    print("Evaluation completed! Check 'evaluation_results' directory for detailed results.")