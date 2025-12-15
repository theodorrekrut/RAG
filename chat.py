import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

qdrant_available = True

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COLLECTION = "docs"
USE_MOCK = os.getenv("USE_MOCK", "0").lower() in ("1", "true", "yes")

if not OPENAI_API_KEY and not USE_MOCK:
    raise RuntimeError(
        "OPENAI_API_KEY not set. Set environment variable OPENAI_API_KEY and retry."
    )

API_BASE = "https://api.openai.com/v1"
HEADERS = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

def client_embeddings(text: str):
    resp = requests.post(
        f"{API_BASE}/embeddings",
        headers=HEADERS,
        json={"model": "text-embedding-3-small", "input": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]
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
    if USE_MOCK:
        return {"choices": [{"message": {"content": f"[MOCK] Antwort zu deiner Frage: {question}"}}]}
    try:
        resp = requests.post(
            f"{API_BASE}/chat/completions",
            headers=HEADERS,
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            raise RuntimeError(
                "OpenAI API rate-limited (429). Try again later or check your API key/quota."
            ) from e
        raise

try:
    resp = chat_completion(prompt)
    print("\n💡 Antwort:")
    try:
        print(resp["choices"][0]["message"]["content"])
    except Exception:
        print(resp)
except RuntimeError as err:
    print(f"[error] {err}")
