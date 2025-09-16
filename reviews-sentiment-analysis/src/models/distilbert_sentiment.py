import torch
import torch.nn as nn
from transformers import (
    DistilBertModel, 
    DistilBertTokenizer, 
    DistilBertConfig,
    PreTrainedModel
)
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class DistilBertForMultilingualSentiment(PreTrainedModel):
    """
    DistilBERT model fine-tuned for multilingual sentiment analysis
    
    This model uses distilbert-base-multilingual-cased as the backbone
    and adds a classification head for 3-class sentiment classification:
    - 0: Negative
    - 1: Neutral  
    - 2: Positive
    """
    
    config_class = DistilBertConfig
    
    def __init__(self, config):
        super().__init__(config)
        
        self.num_labels = 3  # negative, neutral, positive
        self.config = config
        
        # DistilBERT backbone
        self.distilbert = DistilBertModel(config)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout if hasattr(config, 'dropout') else 0.1),
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.num_labels)
        )
        
        # Initialize weights
        self.init_weights()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: Token ids
            attention_mask: Attention mask
            labels: Ground truth labels for training
            
        Returns:
            Dictionary containing loss (if labels provided) and logits
        """
        # Get DistilBERT outputs
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation (first token)
        sequence_output = outputs.last_hidden_state[:, 0]  # Shape: (batch_size, hidden_size)
        
        # Classification
        logits = self.classifier(sequence_output)
        
        result = {"logits": logits}
        
        # Calculate loss if labels are provided
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            result["loss"] = loss
            
        return result
    
    @classmethod
    def from_pretrained_multilingual(cls, cache_dir: Optional[str] = None):
        """
        Load pretrained multilingual DistilBERT and adapt for sentiment analysis
        
        Args:
            cache_dir: Directory to cache the model
            
        Returns:
            Model instance
        """
        model_name = "distilbert-base-multilingual-cased"
        
        # Load configuration
        config = DistilBertConfig.from_pretrained(model_name, cache_dir=cache_dir)
        
        # Create model instance
        model = cls(config)
        
        # Load pretrained DistilBERT weights (only backbone)
        pretrained_model = DistilBertModel.from_pretrained(model_name, cache_dir=cache_dir)
        model.distilbert.load_state_dict(pretrained_model.state_dict())
        
        logger.info(f"Loaded pretrained multilingual DistilBERT from {model_name}")
        logger.info(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        return model

class SentimentTokenizer:
    """
    Tokenizer wrapper for sentiment analysis
    """
    
    def __init__(self, model_name: str = "distilbert-base-multilingual-cased", cache_dir: Optional[str] = None):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.max_length = 512
        
    def __call__(self, texts, **kwargs):
        """
        Tokenize texts for model input
        
        Args:
            texts: List of texts or single text
            **kwargs: Additional arguments
            
        Returns:
            Tokenized inputs
        """
        if isinstance(texts, str):
            texts = [texts]
            
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
    
    def decode(self, token_ids, **kwargs):
        """Decode token ids back to text"""
        return self.tokenizer.decode(token_ids, **kwargs)


class SentimentPredictor:
    """
    Convenient wrapper for making predictions
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.label_names = ["negative", "neutral", "positive"]
        
        # Load model and tokenizer
        if model_path:
            # Load trained model checkpoint
            try:
                # For PyTorch 2.6+ compatibility - trust our own trained models
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                
                # Load pretrained model first
                self.model = DistilBertForMultilingualSentiment.from_pretrained_multilingual()
                
                # Load fine-tuned weights
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    # Fallback for old format
                    self.model = checkpoint
                    
                self.tokenizer = SentimentTokenizer()
            except Exception as e:
                logger.warning(f"Failed to load trained model {model_path}: {e}")
                logger.info("Falling back to pretrained multilingual DistilBERT")
                self.model = DistilBertForMultilingualSentiment.from_pretrained_multilingual()
                self.tokenizer = SentimentTokenizer()
        else:
            # Use pretrained multilingual DistilBERT
            self.model = DistilBertForMultilingualSentiment.from_pretrained_multilingual()
            self.tokenizer = SentimentTokenizer()
            
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, texts, return_probabilities: bool = False):
        """
        Predict sentiment for given texts
        
        Args:
            texts: List of texts or single text
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Predictions (labels and optionally probabilities)
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # Tokenize
        inputs = self.tokenizer(texts)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs["logits"]
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
        
        # Convert to labels
        predicted_labels = [self.label_names[pred.item()] for pred in predictions]
        
        if return_probabilities:
            probs = probabilities.cpu().numpy()
            return predicted_labels, probs
        else:
            return predicted_labels
    
    def predict_single(self, text: str, return_probabilities: bool = False):
        """
        Predict sentiment for a single text
        
        Args:
            text: Input text
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Single prediction
        """
        result = self.predict([text], return_probabilities)
        
        if return_probabilities:
            labels, probs = result
            return labels[0], probs[0]
        else:
            return result[0]

# Test function
def test_model():
    """
    Test the model implementation
    """
    print("Testing DistilBERT Multilingual Sentiment Model...")
    
    # Test model creation
    try:
        model = DistilBertForMultilingualSentiment.from_pretrained_multilingual()
        tokenizer = SentimentTokenizer()
        
        # Test forward pass
        test_texts = [
            "I love this product! It's amazing.",
            "This is okay, nothing special.",
            "Terrible quality, waste of money."
        ]
        
        inputs = tokenizer(test_texts)
        outputs = model(**inputs)
        
        print(f"✓ Model forward pass successful")
        print(f"✓ Input shape: {inputs['input_ids'].shape}")
        print(f"✓ Output logits shape: {outputs['logits'].shape}")
        
        # Test predictor
        predictor = SentimentPredictor()
        predictions = predictor.predict(test_texts, return_probabilities=True)
        
        print(f"✓ Prediction successful")
        print("Sample predictions:")
        for text, (label, probs) in zip(test_texts, zip(predictions[0], predictions[1])):
            print(f"  Text: {text[:50]}...")
            print(f"  Predicted: {label} (confidence: {max(probs):.3f})")
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise
        
    print("All tests passed!")

if __name__ == "__main__":
    test_model()