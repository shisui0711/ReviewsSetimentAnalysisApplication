import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AmazonReviewsDataLoader:
    """
    Data loader for Amazon Reviews Multi dataset
    Handles loading, preprocessing and sentiment label conversion
    """
    
    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = Path(data_dir)
        self.label_mapping = {
            1: 0,  # Negative (1-2 stars)
            2: 0,  # Negative
            3: 1,  # Neutral (3 stars)  
            4: 2,  # Positive (4-5 stars)
            5: 2   # Positive
        }
        self.label_names = ["negative", "neutral", "positive"]
        
    def load_data(self, split: str = "train") -> pd.DataFrame:
        """
        Load data from CSV files
        
        Args:
            split: One of ['train', 'validation', 'test']
            
        Returns:
            DataFrame with loaded data
        """
        file_path = self.data_dir / f"amazon_reviews_multi_{split}.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
        logger.info(f"Loading {split} data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Basic data validation
        required_columns = ['review_title', 'review_body', 'review_id', 'stars']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        logger.info(f"Loaded {len(df)} samples from {split} split")
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data for sentiment analysis
        
        Args:
            df: Raw dataframe from CSV
            
        Returns:
            Processed dataframe with sentiment labels
        """
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Combine title and body for full text
        processed_df['text'] = processed_df['review_title'].fillna('') + ' ' + processed_df['review_body'].fillna('')
        processed_df['text'] = processed_df['text'].str.strip()
        
        # Convert star ratings to sentiment labels (0: negative, 1: neutral, 2: positive)
        processed_df['sentiment'] = processed_df['stars'].map(self.label_mapping)
        
        # Remove rows with missing sentiment labels
        processed_df = processed_df.dropna(subset=['sentiment'])
        processed_df['sentiment'] = processed_df['sentiment'].astype(int)
        
        # Extract language from review_id (format: lang_id)
        processed_df['language'] = processed_df['review_id'].str.split('_').str[0]
        
        # Remove empty or very short texts
        processed_df = processed_df[processed_df['text'].str.len() >= 10]
        
        # Log distribution
        sentiment_dist = processed_df['sentiment'].value_counts().sort_index()
        logger.info("Sentiment distribution:")
        for label, count in sentiment_dist.items():
            logger.info(f"  {self.label_names[label]}: {count} ({count/len(processed_df)*100:.1f}%)")
            
        lang_dist = processed_df['language'].value_counts()
        logger.info(f"Languages found: {list(lang_dist.index)}")
        
        return processed_df[['text', 'sentiment', 'language', 'review_id']]
    
    def get_balanced_sample(self, df: pd.DataFrame, samples_per_class: int = 1000) -> pd.DataFrame:
        """
        Get a balanced sample from the dataset for faster training/testing
        
        Args:
            df: Preprocessed dataframe
            samples_per_class: Number of samples per sentiment class
            
        Returns:
            Balanced dataframe
        """
        balanced_dfs = []
        
        for sentiment in [0, 1, 2]:
            sentiment_df = df[df['sentiment'] == sentiment]
            if len(sentiment_df) >= samples_per_class:
                sampled_df = sentiment_df.sample(n=samples_per_class, random_state=42)
            else:
                sampled_df = sentiment_df
                logger.warning(f"Only {len(sentiment_df)} samples available for {self.label_names[sentiment]}")
            
            balanced_dfs.append(sampled_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
        
        logger.info(f"Created balanced dataset with {len(balanced_df)} samples")
        return balanced_df
    
    def load_all_splits(self, balanced: bool = False, samples_per_class: int = 1000) -> Dict[str, pd.DataFrame]:
        """
        Load and preprocess all data splits
        
        Args:
            balanced: Whether to create balanced samples
            samples_per_class: Number of samples per class if balanced=True
            
        Returns:
            Dictionary containing train, validation, and test dataframes
        """
        splits = {}
        
        for split_name in ['train', 'validation', 'test']:
            try:
                raw_df = self.load_data(split_name)
                processed_df = self.preprocess_data(raw_df)
                
                if balanced and split_name == 'train':
                    processed_df = self.get_balanced_sample(processed_df, samples_per_class)
                
                splits[split_name] = processed_df
                
            except FileNotFoundError:
                logger.warning(f"File for {split_name} split not found, skipping...")
                continue
                
        return splits

def analyze_dataset(data_dir: str = "datasets"):
    """
    Analyze the Amazon Reviews Multi dataset
    """
    loader = AmazonReviewsDataLoader(data_dir)
    
    print("=" * 60)
    print("Amazon Reviews Multi Dataset Analysis")
    print("=" * 60)
    
    try:
        # Load all splits
        all_splits = loader.load_all_splits(balanced=False)
        
        for split_name, df in all_splits.items():
            print(f"\n{split_name.upper()} SPLIT:")
            print(f"- Total samples: {len(df):,}")
            print(f"- Average text length: {df['text'].str.len().mean():.1f} characters")
            
            # Sentiment distribution
            sentiment_counts = df['sentiment'].value_counts().sort_index()
            print("- Sentiment distribution:")
            for sentiment, count in sentiment_counts.items():
                print(f"  * {loader.label_names[sentiment]}: {count:,} ({count/len(df)*100:.1f}%)")
            
            # Language distribution  
            lang_counts = df['language'].value_counts()
            print(f"- Languages: {len(lang_counts)} ({', '.join(lang_counts.head().index.tolist())})")
            
            # Sample texts
            print("- Sample texts:")
            for sentiment in [0, 1, 2]:
                sample = df[df['sentiment'] == sentiment]['text'].iloc[0]
                sample_truncated = sample[:100] + "..." if len(sample) > 100 else sample
                print(f"  * {loader.label_names[sentiment]}: {sample_truncated}")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        logger.error(f"Error analyzing dataset: {e}")
        raise

if __name__ == "__main__":
    analyze_dataset()