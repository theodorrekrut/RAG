import requests
import os
from dotenv import load_dotenv
import ollama

# Load environment variables from .env file
load_dotenv()

qdrant_available = True

# Ollama configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "r1"
COLLECTION = "docs"
USE_MOCK = os.getenv("USE_MOCK", "0").lower() in ("1", "true", "yes")

# Initialize Ollama client
ollama_client = ollama.Client(host=OLLAMA_HOST)

def client_embeddings(text: str):
    """Generate embeddings using Ollama with nomic-embed-text model."""
    response = ollama_client.embed(model=EMBEDDING_MODEL, input=text)
    return response['embeddings'][0]
qdrant = None
if qdrant_available:
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    try:
        r = requests.get(f"{QDRANT_URL}/collections")
        if r.ok:
            print(f"[info] Qdrant REST reachable at {QDRANT_URL}")
        else:
            print(f"[warning] Qdrant REST returned status {r.status_code}")
    except Exception as e:
        print(f"[warning] could not reach Qdrant at {QDRANT_URL}: {e}")
        qdrant_available = False

def embed(text: str):
    return client_embeddings(text)

question = input("❓ Frage: ")

if qdrant_available:
    try:
        qvec = embed(question)
        body = {"vector": qvec, "limit": 5, "with_payload": True}
        r = requests.post(f"{QDRANT_URL}/collections/{COLLECTION}/points/search", json=body, timeout=30)
        if r.ok:
            data = r.json()
            hits = data.get("result") or data.get("data") or []
            texts = []
            for h in hits:
                payload = h.get("payload") or h.get("payload", {})
                if isinstance(payload, dict) and "text" in payload:
                    texts.append(payload["text"])
            context = "\n".join(texts)
        else:
            print(f"[warning] Qdrant search failed: {r.status_code} {r.text}")
            context = ""
    except Exception as e:
        print(f"[warning] Qdrant REST search error: {e}")
        context = ""
else:
    print("[warning] Qdrant not available — proceeding without context from vector DB.")
    context = ""

prompt = f"""
Nutze nur diese Infos:
{context}

Frage:
{question}
"""

def chat_completion(prompt: str):
    """Generate chat completion using Ollama with r1 model."""
    if USE_MOCK:
        return {"choices": [{"message": {"content": f"[MOCK] Response to your query"}}]}
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

try:
    resp = chat_completion(prompt)
    print("\n💡 Antwort:")
    try:
        print(resp["choices"][0]["message"]["content"])
    except Exception:
        print(resp)
except RuntimeError as err:
    print(f"[error] {err}")
