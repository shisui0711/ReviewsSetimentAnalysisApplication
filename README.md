# 🤖 Reviews Sentiment Analysis Application

A beautiful, modern web application for analyzing sentiment in text using advanced AI models. Built with FastAPI backend and Next.js frontend, containerized with Docker for easy deployment.

![Sentiment Analysis Preview](https://via.placeholder.com/800x400/4F46E5/FFFFFF?text=Sentiment+Analysis+Application)

## ✨ Features

### 🎯 Core Functionality
- **Single Text Analysis** - Analyze individual texts with detailed sentiment breakdown
- **Batch Processing** - Process multiple texts simultaneously (up to 50 at once)
- **Real-time Results** - Get instant sentiment classification with confidence scores
- **Multilingual Support** - Supports 10+ languages including English, Spanish, French, German, and more

### 🎨 Beautiful UI
- **Modern Design** - Clean, responsive interface built with Tailwind CSS
- **Smooth Animations** - Framer Motion powered transitions and interactions
- **Dark/Light Mode** - Automatic theme switching based on system preferences
- **Mobile Responsive** - Works perfectly on all device sizes

### 🔧 Technical Features
- **DistilBERT Model** - State-of-the-art transformer model for sentiment analysis
- **RESTful API** - Well-documented FastAPI backend with automatic OpenAPI docs
- **Docker Support** - Complete containerization for easy deployment
- **Health Monitoring** - Built-in health checks and monitoring
- **CORS Enabled** - Proper cross-origin resource sharing configuration

## 🚀 Quick Start

### Prerequisites
- [Docker](https://www.docker.com/get-started) installed and running
- [Git](https://git-scm.com/) for cloning the repository

### 🔥 One-Command Setup

#### For Windows Users:
```bash
# Start development environment
start.bat dev

# Start production environment
start.bat prod
```

#### For Mac/Linux Users:
```bash
# Make script executable
chmod +x start.sh

# Start development environment
./start.sh dev

# Start production environment
./start.sh prod
```

### 🌐 Access Points

**Development Mode:**
- Frontend: http://localhost:3000
- API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

**Production Mode:**
- Application: http://localhost
- API: http://localhost/api
- API Documentation: http://localhost/docs

## 📁 Project Structure

```
ReviewsSetimentAnalysisApplication/
├── reviews-sentiment-analysis/          # FastAPI Backend
│   ├── src/
│   │   ├── api/
│   │   │   └── main.py                 # FastAPI application
│   │   ├── models/
│   │   │   └── distilbert_sentiment.py # DistilBERT model implementation
│   │   ├── data/
│   │   │   └── data_loader.py          # Data loading utilities
│   │   ├── training/
│   │   │   └── trainer.py              # Model training scripts
│   │   └── evaluation/
│   │       └── evaluator.py            # Model evaluation tools
│   ├── requirements.txt                # Python dependencies
│   └── Dockerfile                      # Backend container config
│
├── reviews-analysis-ui/                # Next.js Frontend
│   ├── app/
│   │   ├── page.tsx                    # Main application page
│   │   ├── layout.tsx                  # Application layout
│   │   └── globals.css                 # Global styles
│   ├── components/
│   │   ├── ui/                         # Reusable UI components
│   │   └── SentimentAnalyzer.tsx       # Main analysis component
│   ├── lib/
│   │   ├── api.ts                      # API client
│   │   └── utils.ts                    # Utility functions
│   ├── types/
│   │   └── sentiment.ts                # TypeScript definitions
│   ├── package.json                    # Node.js dependencies
│   └── Dockerfile                      # Frontend container config
│
├── docker-compose.yml                  # Production deployment
├── docker-compose.dev.yml              # Development environment
├── nginx.conf                          # Reverse proxy configuration
├── start.sh                           # Linux/Mac startup script
├── start.bat                          # Windows startup script
└── README.md                          # This file
```

## 🛠️ Development

### Manual Setup (Without Docker)

#### Backend Setup:
```bash
cd reviews-sentiment-analysis

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend Setup:
```bash
cd reviews-analysis-ui

# Install dependencies
pnpm install  # or npm install / yarn install

# Run development server
pnpm dev  # or npm run dev / yarn dev
```

### 🔧 Available Scripts

#### Docker Commands:
```bash
# Development with hot reload
./start.sh dev

# Production deployment
./start.sh prod

# Stop all services
./start.sh stop

# Clean up volumes and containers
./start.sh clean

# View logs
docker-compose logs -f
```

#### Manual Commands:
```bash
# Backend
cd reviews-sentiment-analysis
uvicorn src.api.main:app --reload

# Frontend
cd reviews-analysis-ui
pnpm dev
```

## 📚 API Documentation

The API is fully documented with OpenAPI/Swagger. Access the interactive documentation at:
- Development: http://localhost:8000/docs
- Production: http://localhost/docs

### Key Endpoints:

#### `POST /analyze` - Single Text Analysis
```json
{
  "text": "I love this product! It's amazing!",
  "language": "en"
}
```

#### `POST /analyze/batch` - Batch Analysis
```json
{
  "texts": [
    "This is great!",
    "I hate this.",
    "It's okay, nothing special."
  ],
  "language": "en"
}
```

#### `GET /health` - Health Check
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-01-01T00:00:00"
}
```

#### `GET /model/info` - Model Information
```json
{
  "model_name": "DistilBERT Multilingual Sentiment Analysis",
  "supported_languages": ["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko", "ar"],
  "num_classes": 3,
  "class_names": ["negative", "neutral", "positive"]
}
```

## 🌍 Supported Languages

- **English** (en)
- **Spanish** (es)
- **French** (fr)
- **German** (de)
- **Italian** (it)
- **Portuguese** (pt)
- **Chinese** (zh)
- **Japanese** (ja)
- **Korean** (ko)
- **Arabic** (ar)

## 🎯 Sentiment Classes

The model classifies text into three sentiment categories:

1. **Positive** 😊 - Happy, satisfied, enthusiastic expressions
2. **Negative** 😞 - Angry, disappointed, frustrated expressions  
3. **Neutral** 😐 - Objective, factual, or balanced statements

Each prediction includes:
- **Sentiment Label** - The predicted class
- **Confidence Score** - Overall confidence (0-1)
- **Probability Breakdown** - Individual probabilities for each class
- **Processing Time** - Analysis duration in milliseconds

## 🚢 Deployment

### Production Deployment

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ReviewsSetimentAnalysisApplication
   ```

2. **Configure environment:**
   ```bash
   # Copy and edit environment files
   cp .env.example .env
   ```

3. **Deploy with Docker:**
   ```bash
   ./start.sh prod
   ```

### Cloud Deployment Options

#### Docker Compose on VPS:
```bash
# On your server
git clone <repository-url>
cd ReviewsSetimentAnalysisApplication
./start.sh prod
```

#### Kubernetes:
```bash
# Convert docker-compose to k8s manifests
kompose convert
kubectl apply -f .
```

#### AWS/GCP/Azure:
- Use their container services (ECS, Cloud Run, Container Instances)
- Configure load balancers and auto-scaling
- Set up SSL certificates for HTTPS

## 🔧 Configuration

### Environment Variables

#### Backend (.env):
```bash
PYTHONPATH=/app
PYTHONUNBUFFERED=1
MODEL_PATH=./models/best_model.pt
LOG_LEVEL=INFO
```

#### Frontend (.env.local):
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_TELEMETRY_DISABLED=1
```

### Model Configuration

The application uses DistilBERT by default, but you can:
- Fine-tune on your own data
- Use different pre-trained models
- Adjust confidence thresholds
- Modify supported languages

## 🧪 Testing

### Running Tests:
```bash
# Backend tests
cd reviews-sentiment-analysis
python -m pytest tests/

# Frontend tests
cd reviews-analysis-ui
pnpm test
```

### Load Testing:
```bash
# Test API performance
curl -X POST "http://localhost:8000/analyze/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Test message"] * 50}'
```

## 🐛 Troubleshooting

### Common Issues:

1. **Docker not starting:**
   - Ensure Docker Desktop is running
   - Check Docker daemon status
   - Restart Docker service

2. **Port conflicts:**
   - Change ports in docker-compose.yml
   - Kill processes using ports 3000/8000

3. **Model loading errors:**
   - Check internet connection for model download
   - Verify disk space for model storage
   - Check Python dependencies

4. **Frontend build errors:**
   - Clear Node.js cache: `pnpm store prune`
   - Delete node_modules and reinstall
   - Check Node.js version compatibility

### Getting Help:

1. Check the logs: `docker-compose logs -f`
2. Verify health endpoints: `curl http://localhost:8000/health`
3. Review configuration files
4. Open an issue on GitHub

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** - For the DistilBERT model and Transformers library
- **FastAPI** - For the excellent web framework
- **Next.js** - For the powerful React framework
- **Tailwind CSS** - For the utility-first CSS framework
- **Docker** - For containerization technology

---

Made with ❤️ for sentiment analysis enthusiasts

For questions or support, please [open an issue](https://github.com/your-repo/issues) on GitHub.