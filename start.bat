@echo off
echo 🚀 Starting Sentiment Analysis Application...

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running. Please start Docker and try again.
    exit /b 1
)

REM Parse command line arguments
set COMMAND=%1
if "%COMMAND%"=="" set COMMAND=dev

if "%COMMAND%"=="dev" goto dev
if "%COMMAND%"=="prod" goto prod
if "%COMMAND%"=="stop" goto stop
if "%COMMAND%"=="clean" goto clean
goto usage

:dev
echo 🔧 Starting development environment...
docker-compose -f docker-compose.dev.yml up --build -d
echo.
echo ✅ Development environment started!
echo 🌐 Frontend: http://localhost:3000
echo 🔌 API: http://localhost:8000
echo 📚 API Docs: http://localhost:8000/docs
echo.
echo 📝 To view logs: docker-compose -f docker-compose.dev.yml logs -f
echo 🛑 To stop: start.bat stop
goto end

:prod
echo 🏭 Starting production environment...
docker-compose up --build -d
echo.
echo ✅ Production environment started!
echo 🌐 Application: http://localhost
echo 🔌 API: http://localhost/api
echo 📚 API Docs: http://localhost/docs
echo.
echo 📝 To view logs: docker-compose logs -f
echo 🛑 To stop: start.bat stop
goto end

:stop
echo 🛑 Stopping all services...
docker-compose -f docker-compose.dev.yml down >nul 2>&1
docker-compose down >nul 2>&1
echo ✅ All services stopped!
goto end

:clean
echo 🧹 Cleaning up all services and volumes...
docker-compose -f docker-compose.dev.yml down -v >nul 2>&1
docker-compose down -v >nul 2>&1
docker system prune -f
echo ✅ Cleanup completed!
goto end

:usage
echo Usage: %0 [dev^|prod^|stop^|clean]
echo.
echo Commands:
echo   dev   - Start development environment with hot reload
echo   prod  - Start production environment with Nginx proxy
echo   stop  - Stop all services
echo   clean - Stop services and remove volumes
echo.
exit /b 1

:end