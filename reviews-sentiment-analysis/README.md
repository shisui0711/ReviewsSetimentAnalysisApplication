# Multilingual Sentiment Analysis with DistilBERT

A complete implementation of multilingual sentiment analysis using DistilBERT with transfer learning, deployed via FastAPI.

## 🎯 Overview

This project implements a multilingual sentiment analysis system that can classify text as **negative**, **neutral**, or **positive**. It uses the `distilbert-base-multilingual-cased` model as backbone and fine-tunes it on the Amazon Reviews Multi dataset.

### Key Features

- ✅ **Multilingual Support**: Works with multiple languages including English, Spanish, French, German, Italian, Portuguese, Chinese, Japanese, Korean, Arabic
- ✅ **Transfer Learning**: Fine-tuned from pretrained DistilBERT multilingual model
- ✅ **Fast API Service**: RESTful API with interactive documentation
- ✅ **Comprehensive Evaluation**: Detailed metrics and error analysis
- ✅ **Easy Training Pipeline**: Simple scripts for model training and deployment

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the project (if from git)
git clone <repository_url>
cd multilingual-sentiment-analysis

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Analysis

```bash
# Analyze the Amazon Reviews Multi dataset
python src/data/data_loader.py
```

### 3. Test Pretrained Model

```bash
# Test the multilingual DistilBERT model (without fine-tuning)
python -c "from src.models.distilbert_sentiment import test_model; test_model()"
```

### 4. Start FastAPI Service

```bash
# Start the API server
cd src/api
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Visit http://localhost:8000 for interactive web interface or http://localhost:8000/docs for API documentation.

## 📁 Project Structure

```
multilingual-sentiment-analysis/
├── datasets/                     # Amazon Reviews Multi dataset
│   ├── amazon_reviews_multi_train.csv
│   ├── amazon_reviews_multi_validation.csv
│   └── amazon_reviews_multi_test.csv
├── src/
│   ├── data/                    # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── models/                  # DistilBERT model implementation
│   │   ├── __init__.py
│   │   └── distilbert_sentiment.py
│   ├── training/                # Model training pipeline
│   │   ├── __init__.py
│   │   └── trainer.py
│   ├── evaluation/             # Model evaluation and metrics
│   │   ├── __init__.py
│   │   └── evaluator.py
│   └── api/                    # FastAPI service
│       ├── __init__.py
│       └── main.py
├── models/                     # Saved trained models (created during training)
├── evaluation_results/         # Evaluation reports and plots
├── requirements.txt
└── README.md
```

## 🔧 Usage Examples

### Python API Usage

```python
from src.models.distilbert_sentiment import SentimentPredictor

# Initialize predictor (uses pretrained multilingual DistilBERT)
predictor = SentimentPredictor()

# Analyze single text
text = "This product is amazing! I love it."
sentiment = predictor.predict_single(text)
print(f"Sentiment: {sentiment}")

# Analyze with probabilities
sentiment, probs = predictor.predict_single(text, return_probabilities=True)
print(f"Sentiment: {sentiment}")
print(f"Probabilities: negative={probs[0]:.3f}, neutral={probs[1]:.3f}, positive={probs[2]:.3f}")

# Analyze multiple texts
texts = [
    "Great product, highly recommended!",
    "It's okay, nothing special.",
    "Terrible quality, waste of money."
]
sentiments = predictor.predict(texts)
print("Predicted sentiments:", sentiments)
```

### REST API Usage

```bash
# Analyze single text
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"text": "This product is amazing!"}'

# Analyze batch of texts
curl -X POST "http://localhost:8000/analyze/batch" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Great product!", "Not bad", "Terrible quality"]}'

# Get model information
curl -X GET "http://localhost:8000/model/info"

# Check API health
curl -X GET "http://localhost:8000/health"
```

### Training Custom Model

```python
from src.training.trainer import train_model

# Train on full dataset (may take several hours)
model_path, trainer = train_model(
    epochs=3,
    batch_size=16,
    learning_rate=2e-5,
    balanced=True
)

# Quick training for testing (small subset)
model_path, trainer = train_model(
    epochs=2,
    batch_size=8,
    max_samples=1000,  # Use only 1000 samples for quick testing
    balanced=True
)

print(f"Model saved to: {model_path}")
```

### Model Evaluation

```python
from src.evaluation.evaluator import run_evaluation

# Evaluate pretrained model
results, error_analysis = run_evaluation()

# Evaluate custom trained model
results, error_analysis = run_evaluation(
    model_path="models/best_model_epoch_3_20250106_120000.pt"
)

print(f"Accuracy: {results['accuracy']:.4f}")
print(f"F1 Score: {results['macro_f1']:.4f}")
```

## 🌍 Multilingual Examples

The model supports multiple languages. Here are some examples:

```python
# English
predictor.predict_single("This product is fantastic!")
# → "positive"

# Spanish
predictor.predict_single("Este producto es fantástico!")
# → "positive"

# French
predictor.predict_single("Ce produit est fantastique!")
# → "positive"

# German
predictor.predict_single("Dieses Produkt ist fantastisch!")
# → "positive"

# Portuguese
predictor.predict_single("Este produto é fantástico!")
# → "positive"

# Italian
predictor.predict_single("Questo prodotto è fantastico!")
# → "positive"
```

## 📊 Model Performance

### Dataset Statistics
- **Training samples**: 200,000
- **Validation samples**: 5,000  
- **Test samples**: 5,000
- **Languages**: English (expandable to 100+ languages)
- **Classes**: 3 (negative, neutral, positive)
- **Class distribution**: 40% negative, 20% neutral, 40% positive

### Expected Performance (After Fine-tuning)
- **Accuracy**: ~85-90%
- **F1 Score (macro)**: ~85-90%
- **Inference time**: ~50-100ms per text
- **Model size**: ~250MB

## 🚀 Deployment Options

### 1. Local Development
```bash
# Run API server locally
python src/api/main.py
```

### 2. Docker Deployment
```dockerfile
# Dockerfile example
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3. Cloud Deployment
The API can be deployed on AWS, Google Cloud, Azure, or other cloud platforms using their container services.

## 🔧 Configuration Options

### Training Configuration
```python
# Full training configuration
train_model(
    data_dir="datasets",           # Dataset directory
    model_save_dir="models",       # Where to save models
    epochs=3,                      # Number of training epochs
    batch_size=16,                 # Batch size
    learning_rate=2e-5,            # Learning rate
    max_samples=None,              # None for full dataset
    balanced=True                  # Use balanced sampling
)
```

### API Configuration
```python
# API server configuration
uvicorn.run(
    "src.api.main:app",
    host="0.0.0.0",               # Host address
    port=8000,                    # Port number
    reload=True,                  # Auto-reload on changes
    log_level="info"              # Logging level
)
```

## 📈 Monitoring and Evaluation

### Real-time Monitoring
The API provides endpoints for monitoring:
- `/health` - Health check
- `/model/info` - Model information
- Processing time included in all responses

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-score
- Per-class metrics
- Confusion matrix
- Error analysis
- Confidence statistics

## 🛠️ Advanced Features

### Custom Training
```python
# Train with custom parameters
from src.training.trainer import SentimentTrainer

trainer = SentimentTrainer(model_save_dir="custom_models")
trainer.prepare_data(train_df, val_df, batch_size=32)
trainer.initialize_model(learning_rate=1e-5)
model_path = trainer.train(epochs=5)
```

### Custom Evaluation
```python
from src.evaluation.evaluator import SentimentEvaluator

evaluator = SentimentEvaluator(model_path)
results = evaluator.evaluate_dataset(test_df)
evaluator.plot_confusion_matrix(results['confusion_matrix'])
```

### Batch Processing
```python
# Process large batches efficiently
import pandas as pd

# Load large dataset
df = pd.read_csv("large_reviews.csv")
texts = df['text'].tolist()

# Process in batches
batch_size = 100
all_predictions = []

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i + batch_size]
    batch_predictions = predictor.predict(batch_texts)
    all_predictions.extend(batch_predictions)

# Save results
df['predicted_sentiment'] = all_predictions
df.to_csv("reviews_with_predictions.csv", index=False)
```

## 🔍 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **CUDA out of memory**: Reduce batch size
   ```python
   train_model(batch_size=8)  # Reduce from 16 to 8
   ```

3. **Model not found**: Check model path
   ```python
   # List available models
   import os
   models = [f for f in os.listdir("models") if f.endswith(".pt")]
   print("Available models:", models)
   ```

4. **API connection issues**: Check if server is running
   ```bash
   curl http://localhost:8000/health
   ```

### Performance Optimization

1. **GPU Usage**: Install CUDA-enabled PyTorch for faster training
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Memory Optimization**: Use gradient accumulation for large batches
3. **Inference Optimization**: Use model quantization for production

## 📚 References

- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [Amazon Reviews Multi Dataset](https://huggingface.co/datasets/amazon_reviews_multi)
- [Transformers Library](https://huggingface.co/transformers/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙋 Support

For questions or issues:
1. Check the troubleshooting section
2. Look at existing issues
3. Create a new issue with detailed description