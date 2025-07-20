@echo off
REM Together.ai Multi-Agent Chat Setup Script for Windows

echo 🚀 Setting up Together.ai Multi-Agent Chat System...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo ✅ Python found

REM Create virtual environment
echo 🔧 Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📦 Installing dependencies...
pip install -r requirements.txt

REM Check if .env exists
if not exist ".env" (
    echo 📝 Creating .env file from template...
    copy .env.example .env
    echo ⚠️  Please edit .env file and add your Together.ai API key!
    echo    Get your free API key from: https://api.together.xyz/settings/api-keys
) else (
    echo ✅ .env file already exists
)

echo.
echo 🎉 Setup complete!
echo.
echo 📋 Next steps:
echo 1. Edit .env file and add your Together.ai API key
echo 2. Run: python main.py
echo 3. Open browser to: http://localhost:8000
echo.
echo 💡 Need help? Check README.md for detailed instructions
pause
