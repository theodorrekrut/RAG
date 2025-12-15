"""
Flask web interface for the RAG system.
Provides a chat interface and PDF ingestion capabilities.
Uses Ollama for local LLM operations.
"""
import os
import uuid
import requests
import time
import random
from flask import Flask, render_template, request, jsonify, send_from_directory
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import ollama

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.getenv("PDF_FOLDER", "pdfs")

# Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "r1"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = "docs"
CHUNK_SIZE = 500
BATCH_SIZE = 16
MAX_UPSERT = 500
USE_MOCK = os.getenv("USE_MOCK", "0").lower() in ("1", "true", "yes")

# Initialize Ollama client
ollama_client = ollama.Client(host=OLLAMA_HOST)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def check_qdrant_connection():
    """Check if Qdrant is available."""
    try:
        r = requests.get(f"{QDRANT_URL}/collections", timeout=5)
        return r.ok
    except Exception:
        return False


def ensure_collection(url: str, collection: str, size: int = 768, distance: str = "Cosine"):
    """Ensure that the collection exists in Qdrant. nomic-embed-text uses 768 dimensions."""
    try:
        r = requests.get(f"{url}/collections", timeout=10)
        if r.ok:
            j = r.json()
            cols = []
            if isinstance(j, dict):
                if "result" in j and isinstance(j["result"], dict) and "collections" in j["result"]:
                    cols = j["result"]["collections"]
                elif "collections" in j and isinstance(j["collections"], list):
                    cols = j["collections"]
                elif "result" in j and isinstance(j["result"], list):
                    cols = j["result"]
            names = [c.get("name") if isinstance(c, dict) else c for c in cols]
            if collection in names:
                return True
    except Exception:
        pass
    
    body = {"vectors": {"size": size, "distance": distance}}
    for endpoint in (f"{url}/collections/{collection}/create", f"{url}/collections/{collection}"):
        try:
            r = requests.put(endpoint, json=body, timeout=10)
            if r.ok or r.status_code in (200, 201, 204):
                return True
        except Exception:
            pass
    return False


def embed_batch(texts: list, max_retries: int = 3):
    """Generate embeddings using Ollama with nomic-embed-text model."""
    if USE_MOCK:
        from hashlib import sha256
        vecs = []
        for t in texts:
            h = sha256(t.encode("utf-8") + b"::embed").digest()
            seed = int.from_bytes(h[:8], "big")
            import random as _random
            rnd = _random.Random(seed)
            vec = [rnd.uniform(-1, 1) for _ in range(768)]  # nomic-embed-text uses 768 dimensions
            vecs.append(vec)
        return vecs
    
    for attempt in range(max_retries):
        try:
            # Ollama embed supports batch processing
            response = ollama_client.embed(model=EMBEDDING_MODEL, input=texts)
            return response['embeddings']
        except Exception as e:
            if attempt < max_retries - 1:
                wait = (2 ** attempt) + random.random()
                print(f"[warning] Ollama embedding failed (attempt {attempt+1}): {e}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
                continue
            raise RuntimeError(f"Failed to get embeddings after {max_retries} retries: {e}") from e


def embed(text: str):
    """Embed a single text."""
    return embed_batch([text])[0]


def pdf_to_text(path):
    """Extract text from PDF file."""
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def chat_completion(prompt: str):
    """Generate chat completion using Ollama with r1 model."""
    if USE_MOCK:
        return {"choices": [{"message": {"content": "[MOCK] This is a mock response."}}]}
    try:
        response = ollama_client.chat(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        # Convert Ollama response format to OpenAI-like format for compatibility
        return {
            "choices": [{
                "message": {
                    "content": response['message']['content']
                }
            }]
        }
    except Exception as e:
        raise RuntimeError(f"Ollama chat completion failed: {e}") from e


@app.route('/')
def index():
    """Render the main chat interface."""
    qdrant_status = check_qdrant_connection()
    return render_template('index.html', qdrant_status=qdrant_status)


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests."""
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        # Get context from Qdrant if available
        context = ""
        if check_qdrant_connection():
            try:
                qvec = embed(question)
                body = {"vector": qvec, "limit": 5, "with_payload": True}
                r = requests.post(
                    f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
                    json=body,
                    timeout=30
                )
                if r.ok:
                    result = r.json()
                    hits = result.get("result") or result.get("data") or []
                    texts = []
                    for h in hits:
                        payload = h.get("payload", {})
                        if isinstance(payload, dict) and "text" in payload:
                            texts.append(payload["text"])
                    context = "\n".join(texts)
            except Exception as e:
                print(f"[warning] Qdrant search error: {e}")
        
        # Build prompt
        if context:
            prompt = f"""Use only this information to answer the question:
{context}

Question:
{question}
"""
        else:
            prompt = f"""Question:
{question}

Note: No specific context available. Please provide a general answer.
"""
        
        # Get response from OpenAI
        response = chat_completion(prompt)
        answer = response["choices"][0]["message"]["content"]
        
        return jsonify({
            'answer': answer,
            'has_context': bool(context)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    """Handle PDF upload and ingestion."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are allowed'}), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Ensure collection exists
        if not ensure_collection(QDRANT_URL, COLLECTION):
            return jsonify({'error': 'Could not create collection in Qdrant'}), 500
        
        # Extract text and chunk it
        text = pdf_to_text(filepath)
        chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
        
        # Process chunks in batches
        points = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch_chunks = chunks[i:i+BATCH_SIZE]
            try:
                vectors = embed_batch(batch_chunks)
            except Exception as e:
                print(f"[error] embedding batch failed: {e}")
                vectors = [None] * len(batch_chunks)
            
            for chunk, vec in zip(batch_chunks, vectors):
                if vec is None:
                    continue
                points.append({
                    "id": str(uuid.uuid4()),
                    "vector": vec,
                    "payload": {"text": chunk, "source": filename}
                })
        
        # Upsert to Qdrant
        total_upserted = 0
        for i in range(0, len(points), MAX_UPSERT):
            sub = {"points": points[i:i+MAX_UPSERT]}
            for method in ("post", "put"):
                try:
                    fn = getattr(requests, method)
                    r = fn(f"{QDRANT_URL}/collections/{COLLECTION}/points", json=sub, timeout=60)
                    if r.ok:
                        total_upserted += len(sub['points'])
                        break
                except Exception:
                    pass
        
        return jsonify({
            'success': True,
            'filename': filename,
            'chunks': len(points),
            'upserted': total_upserted
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status')
def status():
    """Get system status."""
    qdrant_connected = check_qdrant_connection()
    
    # Check Ollama connection
    ollama_connected = False
    try:
        # Try to list models to verify Ollama is running
        ollama_client.list()
        ollama_connected = True
    except Exception:
        pass
    
    # Count documents in collection
    doc_count = 0
    if qdrant_connected:
        try:
            r = requests.get(f"{QDRANT_URL}/collections/{COLLECTION}", timeout=5)
            if r.ok:
                data = r.json()
                if "result" in data and "points_count" in data["result"]:
                    doc_count = data["result"]["points_count"]
        except Exception:
            pass
    
    return jsonify({
        'qdrant_connected': qdrant_connected,
        'ollama_connected': ollama_connected,
        'use_mock': USE_MOCK,
        'document_count': doc_count,
        'embedding_model': EMBEDDING_MODEL,
        'chat_model': CHAT_MODEL
    })


if __name__ == '__main__':
    # Check Ollama connection
    ollama_ok = False
    if not USE_MOCK:
        try:
            ollama_client.list()
            ollama_ok = True
        except Exception as e:
            print(f"[WARNING] Ollama not available at {OLLAMA_HOST}: {e}")
            print("[INFO] Make sure Ollama is running: https://ollama.ai")
    
    # Check Qdrant connection
    if not check_qdrant_connection():
        print(f"[WARNING] Qdrant not available at {QDRANT_URL}")
        print("[INFO] Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
    
    print("\n" + "="*60)
    print("RAG Web Interface Starting (Ollama-powered)")
    print("="*60)
    print(f"Ollama Host: {OLLAMA_HOST}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"Chat Model: {CHAT_MODEL}")
    print(f"Qdrant URL: {QDRANT_URL}")
    print(f"Collection: {COLLECTION}")
    print(f"Mock Mode: {USE_MOCK}")
    print("="*60 + "\n")
    
    # Only enable debug mode in development (via environment variable)
    debug_mode = os.getenv("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")
    if debug_mode:
        print("[WARNING] Running in DEBUG mode. Do not use in production!")
    
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)
