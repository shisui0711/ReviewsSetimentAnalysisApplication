#!/bin/bash
# Sentiment Analysis Application Startup Script

echo "🚀 Starting Sentiment Analysis Application..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Function to show usage
show_usage() {
    echo "Usage: $0 [dev|prod|stop|clean]"
    echo ""
    echo "Commands:"
    echo "  dev   - Start development environment with hot reload"
    echo "  prod  - Start production environment with Nginx proxy"
    echo "  stop  - Stop all services"
    echo "  clean - Stop services and remove volumes"
    echo ""
    exit 1
}

# Parse command line arguments
COMMAND=${1:-dev}

case $COMMAND in
    "dev")
        echo "🔧 Starting development environment..."
        docker-compose -f docker-compose.dev.yml up --build -d
        echo ""
        echo "✅ Development environment started!"
        echo "🌐 Frontend: http://localhost:3000"
        echo "🔌 API: http://localhost:8000"
        echo "📚 API Docs: http://localhost:8000/docs"
        echo ""
        echo "📝 To view logs: docker-compose -f docker-compose.dev.yml logs -f"
        echo "🛑 To stop: ./start.sh stop"
        ;;
    
    "prod")
        echo "🏭 Starting production environment..."
        docker-compose up --build -d
        echo ""
        echo "✅ Production environment started!"
        echo "🌐 Application: http://localhost"
        echo "🔌 API: http://localhost/api"
        echo "📚 API Docs: http://localhost/docs"
        echo ""
        echo "📝 To view logs: docker-compose logs -f"
        echo "🛑 To stop: ./start.sh stop"
        ;;
    
    "stop")
        echo "🛑 Stopping all services..."
        docker-compose -f docker-compose.dev.yml down 2>/dev/null
        docker-compose down 2>/dev/null
        echo "✅ All services stopped!"
        ;;
    
    "clean")
        echo "🧹 Cleaning up all services and volumes..."
        docker-compose -f docker-compose.dev.yml down -v 2>/dev/null
        docker-compose down -v 2>/dev/null
        docker system prune -f
        echo "✅ Cleanup completed!"
        ;;
    
    *)
        show_usage
        ;;
esac