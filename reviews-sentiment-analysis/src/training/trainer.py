import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import os
from typing import Dict, List, Optional, Tuple
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from ..models.distilbert_sentiment import DistilBertForMultilingualSentiment, SentimentTokenizer
from ..data.data_loader import AmazonReviewsDataLoader

logger = logging.getLogger(__name__)

class SentimentDataset(Dataset):
    """
    PyTorch Dataset for sentiment analysis
    """
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer: SentimentTokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Tokenize individual text (no batch dimension)
        encoding = self.tokenizer.tokenizer(
            text,
            truncation=True,
            padding=False,  # Don't pad individual items
            max_length=512,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences
    """
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = torch.stack([item['labels'] for item in batch])
    
    # Pad sequences to the same length within the batch
    from torch.nn.utils.rnn import pad_sequence
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': labels
    }

class SentimentTrainer:
    """
    Training pipeline for DistilBERT sentiment analysis model
    """
    
    def __init__(
        self,
        model_save_dir: str = "models",
        device: Optional[str] = None,
        seed: int = 42
    ):
        self.model_save_dir = model_save_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create model directory
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }
        
        logger.info(f"Trainer initialized with device: {self.device}")
    
    def prepare_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        batch_size: int = 16,
        max_samples: Optional[int] = None
    ):
        """
        Prepare data loaders for training
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            batch_size: Batch size for training
            max_samples: Maximum samples to use (for faster experimentation)
        """
        # Sample data if max_samples specified
        if max_samples:
            train_df = train_df.sample(n=min(max_samples, len(train_df)), random_state=self.seed)
            val_df = val_df.sample(n=min(max_samples//5, len(val_df)), random_state=self.seed)
        
        # Initialize tokenizer
        self.tokenizer = SentimentTokenizer()
        
        # Create datasets
        train_dataset = SentimentDataset(
            texts=train_df['text'].tolist(),
            labels=train_df['sentiment'].tolist(),
            tokenizer=self.tokenizer
        )
        
        val_dataset = SentimentDataset(
            texts=val_df['text'].tolist(),
            labels=val_df['sentiment'].tolist(),
            tokenizer=self.tokenizer
        )
        
        # Create data loaders with custom collate function
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for Windows compatibility
            collate_fn=collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        logger.info(f"Data prepared:")
        logger.info(f"  - Train samples: {len(train_dataset)}")
        logger.info(f"  - Validation samples: {len(val_dataset)}")
        logger.info(f"  - Batch size: {batch_size}")
    
    def initialize_model(self, learning_rate: float = 2e-5):
        """
        Initialize model and optimizer
        
        Args:
            learning_rate: Learning rate for training
        """
        # Load pretrained model
        self.model = DistilBertForMultilingualSentiment.from_pretrained_multilingual()
        self.model.to(self.device)
        
        # Setup optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01,
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        
        logger.info(f"Model initialized with learning rate: {learning_rate}")
    
    def train_epoch(self) -> float:
        """
        Train for one epoch
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> Tuple[float, float, Dict]:
        """
        Validate the model
        
        Returns:
            Average validation loss, accuracy, and detailed metrics
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss']
                logits = outputs['logits']
                
                total_loss += loss.item()
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Detailed metrics
        report = classification_report(
            all_labels, 
            all_predictions,
            target_names=['negative', 'neutral', 'positive'],
            output_dict=True
        )
        
        return avg_loss, accuracy, report
    
    def train(
        self,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        warmup_steps: int = 100,
        save_best_model: bool = True
    ):
        """
        Complete training pipeline
        
        Args:
            epochs: Number of epochs to train
            learning_rate: Learning rate
            warmup_steps: Warmup steps for learning rate scheduler
            save_best_model: Whether to save the best model
        """
        if self.model is None:
            self.initialize_model(learning_rate)
        
        # Setup scheduler
        total_steps = len(self.train_loader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        best_val_accuracy = 0
        best_model_path = None
        
        logger.info(f"Starting training for {epochs} epochs...")
        logger.info(f"  - Total training steps: {total_steps}")
        logger.info(f"  - Warmup steps: {warmup_steps}")
        
        for epoch in range(epochs):
            logger.info(f"\n--- Epoch {epoch + 1}/{epochs} ---")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_accuracy, val_report = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            self.history['val_f1'].append(val_report['macro avg']['f1-score'])
            
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Val Accuracy: {val_accuracy:.4f}")
            logger.info(f"Val F1 (macro): {val_report['macro avg']['f1-score']:.4f}")
            
            # Save best model
            if save_best_model and val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_path = self.save_model(f"best_model_epoch_{epoch+1}")
                logger.info(f"New best model saved: {best_model_path}")
        
        # Save final model
        final_model_path = self.save_model("final_model")
        
        # Save training history
        self.save_training_history()
        
        # Plot training history
        self.plot_training_history()
        
        logger.info(f"Training completed!")
        logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
        logger.info(f"Final model saved: {final_model_path}")
        
        return best_model_path if best_model_path else final_model_path
    
    def save_model(self, name: str) -> str:
        """
        Save model and tokenizer
        
        Args:
            name: Model name
            
        Returns:
            Path to saved model
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.model_save_dir, f"{name}_{timestamp}.pt")
        
        # Save model state with PyTorch 2.6+ compatibility
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.config,
            'history': self.history,
            'timestamp': timestamp
        }, model_path, _use_new_zipfile_serialization=False)
        
        return model_path
    
    def save_training_history(self):
        """Save training history to JSON"""
        history_path = os.path.join(self.model_save_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.history['val_accuracy'], label='Val Accuracy')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score plot
        axes[1, 0].plot(self.history['val_f1'], label='Val F1 (macro)')
        axes[1, 0].set_title('Validation F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Remove empty subplot
        fig.delaxes(axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_dir, 'training_history.png'))
        plt.show()

def train_model(
    data_dir: str = "datasets",
    model_save_dir: str = "models", 
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_samples: Optional[int] = None,
    balanced: bool = True
):
    """
    Complete training pipeline
    
    Args:
        data_dir: Directory containing dataset
        model_save_dir: Directory to save models
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        max_samples: Maximum samples for quick testing
        balanced: Whether to use balanced dataset
    """
    # Load data
    data_loader = AmazonReviewsDataLoader(data_dir)
    all_splits = data_loader.load_all_splits(balanced=balanced, samples_per_class=max_samples//3 if max_samples else 1000)
    
    if 'train' not in all_splits or 'validation' not in all_splits:
        raise ValueError("Train and validation splits are required")
    
    # Initialize trainer
    trainer = SentimentTrainer(model_save_dir=model_save_dir)
    
    # Prepare data
    trainer.prepare_data(
        train_df=all_splits['train'],
        val_df=all_splits['validation'],
        batch_size=batch_size,
        max_samples=max_samples
    )
    
    # Train
    model_path = trainer.train(
        epochs=epochs,
        learning_rate=learning_rate
    )
    
    return model_path, trainer

if __name__ == "__main__":
    # Quick training example
    print("Starting sentiment analysis model training...")
    
    model_path, trainer = train_model(
        epochs=2,  # Quick test
        batch_size=8,  # Small batch for testing
        max_samples=1000,  # Small dataset for testing
        balanced=True
    )
    
    print(f"Model trained and saved to: {model_path}")