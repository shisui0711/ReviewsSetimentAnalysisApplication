from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union
import torch
import logging
import os
import json
from datetime import datetime
import uvicorn

# Import our models
from ..models.distilbert_sentiment import SentimentPredictor, DistilBertForMultilingualSentiment, SentimentTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class TextInput(BaseModel):
    text: str = Field(..., description="Text to analyze for sentiment", min_length=1, max_length=1000)
    language: Optional[str] = Field(None, description="Language of the text (optional)")

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyze", min_items=1, max_items=50)
    language: Optional[str] = Field(None, description="Language of the texts (optional)")

class SentimentResult(BaseModel):
    text: str = Field(..., description="Original text")
    sentiment: str = Field(..., description="Predicted sentiment: negative, neutral, or positive")
    confidence: float = Field(..., description="Confidence score (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Probability distribution over all classes")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class BatchSentimentResult(BaseModel):
    results: List[SentimentResult] = Field(..., description="List of sentiment analysis results")
    total_processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    average_processing_time_ms: float = Field(..., description="Average processing time per text")

class ModelInfo(BaseModel):
    model_name: str = Field(..., description="Name of the model")
    model_type: str = Field(..., description="Type of the model")
    supported_languages: List[str] = Field(..., description="List of supported languages")
    num_classes: int = Field(..., description="Number of sentiment classes")
    class_names: List[str] = Field(..., description="Names of sentiment classes")
    model_size_mb: Optional[float] = Field(None, description="Model size in MB")
    last_updated: str = Field(..., description="Last update timestamp")

# Global model instance
predictor: Optional[SentimentPredictor] = None

# FastAPI app
app = FastAPI(
    title="Multilingual Sentiment Analysis API",
    description="DistilBERT-based multilingual sentiment analysis service supporting multiple languages for product reviews and text classification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://127.0.0.1:3000",  # Alternative localhost
        "http://localhost:3001",  # Alternative port
        "*"  # Allow all origins in development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_model(model_path: Optional[str] = None):
    """Load the sentiment analysis model"""
    global predictor
    try:
        if model_path and os.path.exists(model_path):
            # Load fine-tuned model
            logger.info(f"Loading fine-tuned model from {model_path}")
            predictor = SentimentPredictor(model_path=model_path)
        else:
            # Use pretrained multilingual model
            logger.info("Loading pretrained multilingual DistilBERT model")
            predictor = SentimentPredictor()
        
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize the model when the API starts"""
    # Try to load the best available model
    model_dir = "models"
    best_model = None
    
    if os.path.exists(model_dir):
        # Look for the best model file
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
        if model_files:
            # Sort by modification time, get the latest
            model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
            best_model = os.path.join(model_dir, model_files[0])
    
    load_model(best_model)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with HTML interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Multilingual Sentiment Analysis API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { background: #f5f5f5; padding: 20px; border-radius: 10px; }
            .endpoint { background: white; margin: 10px 0; padding: 15px; border-radius: 5px; }
            .method { background: #007acc; color: white; padding: 5px 10px; border-radius: 3px; font-weight: bold; }
            .post { background: #49cc90; }
            textarea { width: 100%; height: 100px; margin: 10px 0; }
            button { background: #007acc; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #005999; }
            .result { background: #e8f4fd; padding: 10px; margin: 10px 0; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Multilingual Sentiment Analysis API</h1>
            <p>DistilBERT-based sentiment analysis supporting multiple languages</p>
            
            <div class="endpoint">
                <h3><span class="method">GET</span> /health</h3>
                <p>Check API health status</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method post">POST</span> /analyze</h3>
                <p>Analyze sentiment of a single text</p>
                <div>
                    <h4>Try it out:</h4>
                    <textarea id="singleText" placeholder="Enter text to analyze...">This product is amazing! I love it.</textarea>
                    <br>
                    <button onclick="analyzeSingle()">Analyze Sentiment</button>
                    <div id="singleResult" class="result" style="display:none;"></div>
                </div>
            </div>
            
            <div class="endpoint">
                <h3><span class="method post">POST</span> /analyze/batch</h3>
                <p>Analyze sentiment of multiple texts</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">GET</span> /model/info</h3>
                <p>Get model information and capabilities</p>
            </div>
            
            <p><strong>Documentation:</strong> 
                <a href="/docs">Swagger UI</a> | 
                <a href="/redoc">ReDoc</a>
            </p>
        </div>
        
        <script>
        async function analyzeSingle() {
            const text = document.getElementById('singleText').value;
            const resultDiv = document.getElementById('singleResult');
            
            if (!text.trim()) {
                alert('Please enter some text to analyze');
                return;
            }
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text })
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    resultDiv.innerHTML = `
                        <h4>Result:</h4>
                        <p><strong>Sentiment:</strong> ${result.sentiment}</p>
                        <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                        <p><strong>Processing Time:</strong> ${result.processing_time_ms.toFixed(2)}ms</p>
                    `;
                    resultDiv.style.display = 'block';
                } else {
                    resultDiv.innerHTML = `<p style="color:red;">Error: ${result.detail}</p>`;
                    resultDiv.style.display = 'block';
                }
            } catch (error) {
                resultDiv.innerHTML = `<p style="color:red;">Error: ${error.message}</p>`;
                resultDiv.style.display = 'block';
            }
        }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global predictor
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/analyze", response_model=SentimentResult)
async def analyze_sentiment(input_data: TextInput):
    """
    Analyze sentiment of a single text
    
    Returns sentiment classification (negative/neutral/positive) with confidence scores
    """
    global predictor
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = datetime.now()
        
        # Get prediction with probabilities
        sentiment, probabilities = predictor.predict_single(
            input_data.text, 
            return_probabilities=True
        )
        
        end_time = datetime.now()
        processing_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Convert probabilities to dict
        prob_dict = {
            "negative": float(probabilities[0]),
            "neutral": float(probabilities[1]),
            "positive": float(probabilities[2])
        }
        
        # Get confidence (max probability)
        confidence = float(max(probabilities))
        
        return SentimentResult(
            text=input_data.text,
            sentiment=sentiment,
            confidence=confidence,
            probabilities=prob_dict,
            processing_time_ms=processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/batch", response_model=BatchSentimentResult)
async def analyze_batch_sentiment(input_data: BatchTextInput):
    """
    Analyze sentiment of multiple texts in batch
    
    More efficient for processing multiple texts at once
    """
    global predictor
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = datetime.now()
        
        # Get predictions with probabilities
        sentiments, all_probabilities = predictor.predict(
            input_data.texts, 
            return_probabilities=True
        )
        
        # Process results
        results = []
        for i, (text, sentiment, probabilities) in enumerate(zip(input_data.texts, sentiments, all_probabilities)):
            prob_dict = {
                "negative": float(probabilities[0]),
                "neutral": float(probabilities[1]),
                "positive": float(probabilities[2])
            }
            
            confidence = float(max(probabilities))
            
            results.append(SentimentResult(
                text=text,
                sentiment=sentiment,
                confidence=confidence,
                probabilities=prob_dict,
                processing_time_ms=0  # Will be calculated per batch
            ))
        
        end_time = datetime.now()
        total_processing_time_ms = (end_time - start_time).total_seconds() * 1000
        avg_processing_time_ms = total_processing_time_ms / len(input_data.texts)
        
        # Update individual processing times
        for result in results:
            result.processing_time_ms = avg_processing_time_ms
        
        return BatchSentimentResult(
            results=results,
            total_processing_time_ms=total_processing_time_ms,
            average_processing_time_ms=avg_processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"Error analyzing batch sentiment: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """
    Get information about the loaded model
    """
    global predictor
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Calculate model size (approximate)
        model_params = sum(p.numel() for p in predictor.model.parameters())
        model_size_mb = (model_params * 4) / (1024 * 1024)  # Assuming float32
        
        return ModelInfo(
            model_name="DistilBERT Multilingual Sentiment Analysis",
            model_type="distilbert-base-multilingual-cased",
            supported_languages=["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko", "ar"],  # Major languages
            num_classes=3,
            class_names=["negative", "neutral", "positive"],
            model_size_mb=round(model_size_mb, 2),
            last_updated=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@app.post("/model/reload")
async def reload_model(model_path: Optional[str] = None):
    """
    Reload the model (useful for updating to a newly trained model)
    """
    try:
        load_model(model_path)
        return {
            "status": "success",
            "message": "Model reloaded successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )