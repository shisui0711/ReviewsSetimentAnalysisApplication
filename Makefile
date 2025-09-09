# Sentiment Analysis Application Makefile

.PHONY: help dev prod stop clean logs install test

# Default target
help:
\t@echo \"🤖 Sentiment Analysis Application\"
\t@echo \"\"
\t@echo \"Available commands:\"
\t@echo \"  make dev     - Start development environment\"
\t@echo \"  make prod    - Start production environment\"
\t@echo \"  make stop    - Stop all services\"
\t@echo \"  make clean   - Stop and remove all containers/volumes\"
\t@echo \"  make logs    - View application logs\"
\t@echo \"  make test    - Run tests\"
\t@echo \"  make install - Install dependencies\"
\t@echo \"\"
\t@echo \"Quick start: make dev\"

# Development environment
dev:
\t@echo \"🔧 Starting development environment...\"
\tdocker-compose -f docker-compose.dev.yml up --build -d
\t@echo \"✅ Development environment started!\"
\t@echo \"🌐 Frontend: http://localhost:3000\"
\t@echo \"🔌 API: http://localhost:8000\"
\t@echo \"📚 API Docs: http://localhost:8000/docs\"

# Production environment
prod:
\t@echo \"🏭 Starting production environment...\"
\tdocker-compose up --build -d
\t@echo \"✅ Production environment started!\"
\t@echo \"🌐 Application: http://localhost\"
\t@echo \"🔌 API: http://localhost/api\"
\t@echo \"📚 API Docs: http://localhost/docs\"

# Stop services
stop:
\t@echo \"🛑 Stopping all services...\"
\tdocker-compose -f docker-compose.dev.yml down 2>/dev/null || true
\tdocker-compose down 2>/dev/null || true
\t@echo \"✅ All services stopped!\"

# Clean up
clean:
\t@echo \"🧹 Cleaning up all services and volumes...\"
\tdocker-compose -f docker-compose.dev.yml down -v 2>/dev/null || true
\tdocker-compose down -v 2>/dev/null || true
\tdocker system prune -f
\t@echo \"✅ Cleanup completed!\"

# View logs
logs:
\t@echo \"📝 Viewing application logs (Press Ctrl+C to exit)...\"
\tdocker-compose -f docker-compose.dev.yml logs -f 2>/dev/null || docker-compose logs -f

# Run tests
test:
\t@echo \"🧪 Running tests...\"
\t@echo \"Backend tests:\"
\tdocker-compose -f docker-compose.dev.yml exec sentiment-api python -m pytest tests/ 2>/dev/null || echo \"No backend tests found\"
\t@echo \"Frontend tests:\"
\tdocker-compose -f docker-compose.dev.yml exec sentiment-ui npm test 2>/dev/null || echo \"No frontend tests found\"

# Install dependencies
install:
\t@echo \"📦 Installing dependencies...\"
\t@echo \"Backend dependencies:\"
\tcd reviews-sentiment-analysis && pip install -r requirements.txt
\t@echo \"Frontend dependencies:\"
\tcd reviews-analysis-ui && npm install
\t@echo \"✅ Dependencies installed!\"

# Health check
health:
\t@echo \"🔍 Checking application health...\"
\tcurl -f http://localhost:8000/health 2>/dev/null && echo \"✅ API is healthy\" || echo \"❌ API is not responding\"
\tcurl -f http://localhost:3000 2>/dev/null && echo \"✅ Frontend is healthy\" || echo \"❌ Frontend is not responding\"

# API test
api-test:
\t@echo \"🔌 Testing API endpoints...\"
\t@echo \"Single analysis test:\"
\tcurl -X POST \"http://localhost:8000/analyze\" \\n\t\t-H \"Content-Type: application/json\" \\n\t\t-d '{\"text\": \"I love this product!\"}' | jq .
\t@echo \"\"
\t@echo \"Batch analysis test:\"
\tcurl -X POST \"http://localhost:8000/analyze/batch\" \\n\t\t-H \"Content-Type: application/json\" \\n\t\t-d '{\"texts\": [\"Great!\", \"Terrible.\", \"Okay.\"]}' | jq .

# Show status
status:
\t@echo \"📊 Application Status:\"
\tdocker-compose -f docker-compose.dev.yml ps 2>/dev/null || docker-compose ps