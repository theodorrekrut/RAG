# RAG - Retrieval Augmented Generation System

A complete RAG (Retrieval Augmented Generation) system with a modern web interface. Upload PDF documents, ingest them into a vector database (Qdrant), and ask questions that are answered using context from your documents.

**Powered by Ollama for 100% local, private AI processing - no API keys, no cloud services, no internet required.**

## Features

- 📄 **PDF Upload & Ingestion**: Upload PDF documents through the web interface
- 🔍 **Vector Search**: Uses Qdrant vector database for semantic search
- 💬 **Chat Interface**: Beautiful web UI for asking questions
- 🤖 **AI-Powered Answers**: Uses local LLMs via Ollama (r1 for reasoning)
- 🔐 **100% Local & Private**: All processing happens on your machine
- 🎯 **Real-time Status**: See system status and document count
- 🚫 **No API Keys Required**: Everything runs locally

## Architecture

- **Embedding Model**: `nomic-embed-text` (768 dimensions) - for converting text to vectors
- **Chat Model**: `r1` - for reasoning and generating answers
- **Vector Database**: Qdrant - for storing and searching document embeddings
- **LLM Server**: Ollama - local LLM inference server

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai) installed and running locally
- Qdrant vector database running locally

## Quick Start

### 1. Install Ollama

Download and install Ollama from https://ollama.ai

Then pull the required models:

```bash
ollama pull nomic-embed-text
ollama pull r1
```

Verify Ollama is running:
```bash
curl http://localhost:11434
```

### 2. Install Qdrant (Docker)

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

Or download and run Qdrant locally from https://qdrant.tech/documentation/quick-start/

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

The default configuration should work for local development:

```
OLLAMA_HOST=http://localhost:11434
QDRANT_URL=http://localhost:6333
```

### 5. Run the Web Interface

```bash
python web_app.py
```

Open your browser and go to: http://localhost:5000

## Usage

### Web Interface (Recommended)

1. Start the web application: `python web_app.py`
2. Upload PDF documents using the sidebar
3. Ask questions in the chat interface
4. The system will search for relevant context and provide AI-generated answers

### Command Line Interface

**Ingest PDFs:**
```bash
# Place PDF files in the 'pdfs' folder
mkdir -p pdfs
cp your-document.pdf pdfs/

# Run ingestion
python ingest_pdf.py
```

**Ask Questions:**
```bash
python chat.py
```

## Project Structure

```
RAG/
├── web_app.py          # Flask web interface (main application)
├── ingest_pdf.py       # PDF ingestion script
├── chat.py             # Command-line chat interface
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variables template
├── .gitignore          # Git ignore file
├── templates/
│   └── index.html      # Web UI template
├── pdfs/               # PDF documents folder (created automatically)
└── README.md           # This file
```

## Configuration

Environment variables (set in `.env` file):

- `OLLAMA_HOST`: Ollama server URL (default: http://localhost:11434)
- `QDRANT_URL`: Qdrant server URL (default: http://localhost:6333)
- `PDF_FOLDER`: Folder containing PDFs (default: pdfs)
- `USE_MOCK`: Use mock mode for testing without Ollama (default: 0)

## Troubleshooting

### Ollama Connection Issues

Make sure Ollama is running and models are available:
```bash
curl http://localhost:11434
ollama list
```

If models are missing:
```bash
ollama pull nomic-embed-text
ollama pull r1
```

### Qdrant Connection Issues

Make sure Qdrant is running:
```bash
curl http://localhost:6333/collections
```

### PDF Processing Issues

- Ensure PDFs are not encrypted or password-protected
- Try smaller PDF files first
- Check the console for detailed error messages

## Development

### Running in Mock Mode

For testing without Ollama:

```bash
USE_MOCK=1 python web_app.py
```

### Customization

- **Chunk Size**: Modify `CHUNK_SIZE` in `web_app.py` or `ingest_pdf.py`
- **Collection Name**: Change `COLLECTION` variable
- **Models**: Update `EMBEDDING_MODEL` and `CHAT_MODEL` variables to use different Ollama models

## Security & Privacy

- **100% Local**: All data stays on your machine
- **No API Keys**: No external services or API keys required
- **Private**: Your documents and queries never leave your computer
- The `.gitignore` file is configured to exclude sensitive files

## License

This project is provided as-is for educational and development purposes.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the console logs for error messages
3. Ensure all prerequisites are properly installed