# Quick Start Guide

## Setup in 3 Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file:
```bash
cp .env.example .env
```

Edit `.env` and set your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
QDRANT_URL=http://localhost:6333
```

### 3. Run the Web Interface
```bash
python web_app.py
```

Or use the startup script:
```bash
./run_web.sh
```

Then open http://localhost:5000 in your browser!

## Using Your Local Qdrant Server

If you already have Qdrant running locally, the system will automatically connect to it at the URL specified in your `.env` file (default: `http://localhost:6333`).

**Custom Qdrant URL:**
```bash
# In .env file
QDRANT_URL=http://192.168.0.241:6333
```

Or set it when running:
```bash
QDRANT_URL=http://192.168.0.241:6333 python web_app.py
```

## Using Without Qdrant (Testing Mode)

If you don't have Qdrant running yet, you can still test the interface in mock mode:

```bash
USE_MOCK=1 python web_app.py
```

## Workflow

1. **Upload PDFs**: Use the web interface sidebar to upload PDF documents
2. **Ask Questions**: Type questions in the chat interface
3. **Get Answers**: The system searches your documents and provides AI-generated answers

## Command Line Tools

**Ingest PDFs manually:**
```bash
# Put PDFs in the pdfs/ folder
cp your-document.pdf pdfs/

# Run ingestion
python ingest_pdf.py
```

**Chat from terminal:**
```bash
python chat.py
```

## Troubleshooting

### Can't connect to Qdrant?
- Make sure Qdrant is running: `docker run -p 6333:6333 qdrant/qdrant`
- Check the URL in your `.env` file
- Test with: `curl http://localhost:6333/collections`

### OpenAI API errors?
- Verify your API key is correct
- Check your API quota at https://platform.openai.com/usage
- Use `USE_MOCK=1` for testing without API calls

### PDF upload fails?
- Make sure the PDF is not password-protected
- Try smaller files first
- Check the browser console for error messages

## Next Steps

- Add your PDF documents through the web interface
- Customize chunk size in `web_app.py` if needed
- Set up production deployment with proper WSGI server (gunicorn, uwsgi)
- Configure persistent Qdrant storage

Enjoy your RAG system! 🚀
