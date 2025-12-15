# RAG - Retrieval Augmented Generation System

A complete RAG (Retrieval Augmented Generation) system with a modern web interface. Upload PDF documents, ingest them into a vector database (Qdrant), and ask questions that are answered using context from your documents.

## Features

- 📄 **PDF Upload & Ingestion**: Upload PDF documents through the web interface
- 🔍 **Vector Search**: Uses Qdrant vector database for semantic search
- 💬 **Chat Interface**: Beautiful web UI for asking questions
- 🤖 **AI-Powered Answers**: Uses OpenAI's GPT-4 to generate contextual answers
- 🎯 **Real-time Status**: See system status and document count
- 🔒 **Secure**: No hardcoded API keys, uses environment variables

## Prerequisites

- Python 3.8+
- Qdrant vector database running locally
- OpenAI API key

## Quick Start

### 1. Install Qdrant (Docker)

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

Or download and run Qdrant locally from https://qdrant.tech/documentation/quick-start/

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=your-openai-api-key-here
QDRANT_URL=http://localhost:6333
```

### 4. Run the Web Interface

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

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `QDRANT_URL`: Qdrant server URL (default: http://localhost:6333)
- `PDF_FOLDER`: Folder containing PDFs (default: pdfs)
- `USE_MOCK`: Use mock mode for testing without API calls (default: 0)

## Troubleshooting

### Qdrant Connection Issues

Make sure Qdrant is running:
```bash
curl http://localhost:6333/collections
```

### OpenAI API Issues

- Verify your API key is correct
- Check your API quota and billing status
- Enable mock mode for testing: `USE_MOCK=1`

### PDF Processing Issues

- Ensure PDFs are not encrypted or password-protected
- Try smaller PDF files first
- Check the console for detailed error messages

## Development

### Running in Mock Mode

For testing without API calls:

```bash
USE_MOCK=1 python web_app.py
```

### Customization

- **Chunk Size**: Modify `CHUNK_SIZE` in `web_app.py` or `ingest_pdf.py`
- **Collection Name**: Change `COLLECTION` variable
- **Model**: Update the model name in OpenAI API calls

## Security Notes

- Never commit your `.env` file or API keys to version control
- The `.gitignore` file is configured to exclude sensitive files
- API keys are loaded from environment variables only

## License

This project is provided as-is for educational and development purposes.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the console logs for error messages
3. Ensure all prerequisites are properly installed