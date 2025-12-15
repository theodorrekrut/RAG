import os
import uuid
import requests
import time
import random
from PyPDF2 import PdfReader  

# ------------------- CONFIG -------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PDF_FOLDER = "pdfs"
COLLECTION = "docs"
CHUNK_SIZE = 500
# ---------------------------------------------

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Export it and retry.")

USE_MOCK = os.getenv("USE_MOCK", "0").lower() in ("1", "true", "yes")

API_BASE = "https://api.openai.com/v1"
HEADERS = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

QDRANT_URL = os.getenv("QDRANT_URL", "http://192.168.0.241:6333")


if not USE_MOCK and not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Export it and retry.")

def ensure_collection(url: str, collection: str, size: int = 1536, distance: str = "Cosine"):
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

def embed_batch(texts: list, max_retries: int = 5):
    """Call OpenAI embeddings API with a batch of texts and retry on 429/5xx."""
    if USE_MOCK:
        
        from hashlib import sha256
        vecs = []
        for t in texts:
            h = sha256(t.encode("utf-8") + b"::embed").digest()
            
            seed = int.from_bytes(h[:8], "big")
            import random as _random
            rnd = _random.Random(seed)
            vec = [rnd.uniform(-1, 1) for _ in range(1536)]
            vecs.append(vec)
        return vecs
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{API_BASE}/embeddings",
                headers=HEADERS,
                json={"model": "text-embedding-3-small", "input": texts},
                timeout=60,
            )
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                raise requests.exceptions.HTTPError(resp.text, response=resp)
            resp.raise_for_status()
            data = resp.json()
            return [d["embedding"] for d in data["data"]]
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                wait = (2 ** attempt) + random.random()
                print(f"[warn] rate limited by OpenAI (429). Backing off {wait:.1f}s (attempt {attempt+1})")
                time.sleep(wait)
                continue
            if e.response is not None and 500 <= e.response.status_code < 600:
                wait = (2 ** attempt) + random.random()
                print(f"[warn] server error {e.response.status_code}. Backing off {wait:.1f}s (attempt {attempt+1})")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Failed to get embeddings after retries")

def embed(text: str):
    return embed_batch([text])[0]

def pdf_to_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

points = []

ensure_collection(QDRANT_URL, COLLECTION, size=1536, distance="Cosine")


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
