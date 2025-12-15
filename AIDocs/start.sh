#!/bin/bash
# Quick start script for Context-Aware RAG Agent

echo "🚀 Starting Context-Aware RAG Agent..."
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please create one first:"
    echo "   python -m venv .venv"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Check if dependencies are installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found"
    echo "   Please create .env with your GOOGLE_API_KEY"
    echo ""
fi

# Check for Tesseract
if ! command -v tesseract &> /dev/null; then
    echo "⚠️  Warning: Tesseract OCR not found"
    echo "   Install with: brew install tesseract"
    echo ""
fi

# Start Streamlit app
echo "✨ Launching Streamlit app..."
echo "   URL: http://localhost:8501"
echo ""
streamlit run src/interface/app.py
