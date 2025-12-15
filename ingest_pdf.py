import os
import uuid
import requests
import time
import random
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import ollama

# Load environment variables from .env file
load_dotenv()

# ------------------- CONFIG -------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = "nomic-embed-text"
PDF_FOLDER = os.getenv("PDF_FOLDER", "pdfs")
COLLECTION = "docs"
CHUNK_SIZE = 500
USE_MOCK = os.getenv("USE_MOCK", "0").lower() in ("1", "true", "yes")
# ---------------------------------------------

# Initialize Ollama client
ollama_client = ollama.Client(host=OLLAMA_HOST)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

def ensure_collection(url: str, collection: str, size: int = 768, distance: str = "Cosine"):
    """Ensure collection exists in Qdrant. nomic-embed-text uses 768 dimensions."""
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
                print(f"[info] collection '{collection}' already exists")
                return True
    except Exception:
        pass
    body = {"vectors": {"size": size, "distance": distance}}
    for endpoint in (f"{url}/collections/{collection}/create", f"{url}/collections/{collection}"):
        try:
            r = requests.put(endpoint, json=body, timeout=10)
            if r.ok or r.status_code in (200, 201, 204):
                print(f"[info] created collection via {endpoint}")
                return True
        except Exception:
            pass
    raise RuntimeError(f"Could not create collection '{collection}' at {url}")

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
                print(f"[warn] Ollama embedding failed (attempt {attempt+1}): {e}. Backing off {wait:.1f}s")
                time.sleep(wait)
                continue
            raise RuntimeError(f"Failed to get embeddings after {max_retries} retries: {e}") from e

def embed(text: str):
    return embed_batch([text])[0]

def pdf_to_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

points = []

# Create PDF folder if it doesn't exist
if not os.path.exists(PDF_FOLDER):
    os.makedirs(PDF_FOLDER)
    print(f"[info] Created {PDF_FOLDER} folder. Please add PDF files to ingest.")

# Check if folder has any PDFs
pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
if not pdf_files:
    print(f"[warning] No PDF files found in {PDF_FOLDER} folder.")
    print(f"[info] Add PDF files to the {PDF_FOLDER} folder and run this script again.")
    exit(0)

# Check Ollama connection (skip in mock mode)
if not USE_MOCK:
    try:
        ollama_client.list()
        print(f"[info] Ollama connection verified at {OLLAMA_HOST}")
    except Exception as e:
        print(f"[error] Cannot connect to Ollama at {OLLAMA_HOST}: {e}")
        print("[error] Make sure Ollama is running: https://ollama.ai")
        print("[info] Alternatively, use USE_MOCK=1 for testing without Ollama")
        exit(1)

# Try to create collection in Qdrant (nomic-embed-text uses 768 dimensions)
try:
    if not ensure_collection(QDRANT_URL, COLLECTION, size=768, distance="Cosine"):
        print(f"[error] Could not create collection '{COLLECTION}' at {QDRANT_URL}")
        print("[error] Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
        exit(1)
except Exception as e:
    print(f"[error] Failed to connect to Qdrant at {QDRANT_URL}: {e}")
    print("[error] Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
    exit(1)

BATCH_SIZE = 16
for filename in os.listdir(PDF_FOLDER):
    if filename.lower().endswith(".pdf"):
        path = os.path.join(PDF_FOLDER, filename)
        text = pdf_to_text(path)
        chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
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

MAX_UPSERT = 500
for i in range(0, len(points), MAX_UPSERT):
    sub = {"points": points[i:i+MAX_UPSERT]}
    success = False
    for method in ("post", "put"):
        try:
            fn = getattr(requests, method)
            r = fn(f"{QDRANT_URL}/collections/{COLLECTION}/points", json=sub, timeout=60)
            if r.ok:
                print(f"✅ Upserted {len(sub['points'])} points (via {method.upper()})")
                success = True
                break
            else:
                print(f"[warning] upsert via {method.upper()} returned {r.status_code}: {r.text}")
        except Exception as e:
            print(f"[warning] upsert via {method.upper()} failed: {e}")
    if not success:
        print(f"[error] failed to upsert batch starting at {i}")
        break
print(f"✅ Fertig, {len(points)} Chunks gespeichert")
