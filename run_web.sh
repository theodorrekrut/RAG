#!/bin/bash
# Simple startup script for the RAG web interface

echo "======================================"
echo "RAG Web Interface Startup Script"
echo "======================================"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found!"
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "✅ Created .env file"
    echo "⚠️  Please edit .env and add your OPENAI_API_KEY"
    echo ""
fi

# Check if pdfs folder exists
if [ ! -d pdfs ]; then
    echo "📁 Creating pdfs folder..."
    mkdir -p pdfs
    echo "✅ Created pdfs folder"
    echo ""
fi

# Check if Python dependencies are installed
echo "📦 Checking dependencies..."
MISSING_DEPS=0
for dep in flask requests PyPDF2 dotenv; do
    if ! python3 -c "import $dep" 2>/dev/null; then
        MISSING_DEPS=1
        break
    fi
done

if [ $MISSING_DEPS -eq 1 ]; then
    echo "⚠️  Missing dependencies detected"
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
else
    echo "✅ All dependencies are installed"
fi
echo ""

# Check if Qdrant is accessible
echo "🔍 Checking Qdrant connection..."
if curl -s http://localhost:6333/collections > /dev/null 2>&1; then
    echo "✅ Qdrant is running at http://localhost:6333"
else
    echo "⚠️  Qdrant not accessible at http://localhost:6333"
    echo "   Start Qdrant with: docker run -p 6333:6333 qdrant/qdrant"
    echo "   The app will still work in mock mode or without vector search"
fi

echo ""
echo "======================================"
echo "Starting web interface on port 5000..."
echo "======================================"
echo ""

# Start the web application
python3 web_app.py
