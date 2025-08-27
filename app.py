import os
import io
import json
import time
import hashlib
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
import requests
import chromadb
import pypdf
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline


import streamlit as st

# --- Optional dependencies: handled gracefully if missing ---
MISSING_DEPS = []
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",   # Use DuckDB instead of SQLite
        persist_directory=".chromadb"       # Optional: store DB locally
    ))
except Exception:
    chromadb = None
    Settings = None
    embedding_functions = None
    client = None
    MISSING_DEPS.append("chromadb")

try:
    import pypdf
except Exception:
    pypdf = None
    MISSING_DEPS.append("pypdf")

try:
    import torch
except Exception:
    torch = None
    MISSING_DEPS.append("torch")

try:
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
except Exception:
    AutoModel = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    pipeline = None
    MISSING_DEPS.append("transformers")


# -------------------------------
# Minimal HuggingFace Embedding Function
# -------------------------------
class HFEmbeddingFunction:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        if not torch or not AutoTokenizer or not AutoModel:
            raise ImportError("torch and transformers are required for HuggingFace embeddings")
        
        # Cache the model in session state to avoid reloading
        cache_key = f"hf_model_{model_name}"
        if cache_key not in st.session_state:
            st.write(f"Loading embedding model: {model_name}")
            st.session_state[cache_key] = {
                'tokenizer': AutoTokenizer.from_pretrained(model_name),
                'model': AutoModel.from_pretrained(model_name)
            }
        
        self.tokenizer = st.session_state[cache_key]['tokenizer']
        self.model = st.session_state[cache_key]['model']
        self.model.eval()  # Set to evaluation mode

    def mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling with attention mask for better embeddings"""
        token_embeddings = model_output[0]  # First element contains token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def __call__(self, texts: List[str]) -> List[List[float]]:
        # Process in smaller batches to avoid memory issues
        batch_size = 8
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize with proper truncation
            inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512,  # Reasonable limit
                return_tensors="pt"
            )
            
            with torch.no_grad():
                model_output = self.model(**inputs)
                
            # Apply mean pooling with attention mask
            embeddings = self.mean_pooling(model_output, inputs['attention_mask'])
            
            # Normalize embeddings (important for similarity search)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.extend(embeddings.cpu().numpy().tolist())
        
        return all_embeddings


# -------------------------------
# LLM Call Function (HuggingFace API only)
# -------------------------------
# Replace your call_llm function with this working version:

def call_llm(messages: List[Dict[str, str]]) -> str:
    """Call HuggingFace Inference Providers API using HF_API_TOKEN."""

    llm_model = st.session_state.get("llm_model_name", "meta-llama/Llama-3.1-8B-Instruct")
    temperature = st.session_state.get("temperature", 0.7)
    max_length = st.session_state.get("max_response_length", 200)

    # HF API token
    hf_token = os.getenv("HF_API_TOKEN", st.session_state.get("hf_api_token", ""))
    if not hf_token:
        return "(HF_API_TOKEN not set; cannot generate response)"

    # Use the HuggingFace Inference Providers API (OpenAI-compatible endpoint)
    API_URL = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": llm_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_length,
        "stream": False
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        
        # Handle common HTTP errors
        if response.status_code == 429:
            return "⏳ Rate limit exceeded. Please try again in a moment."
        elif response.status_code == 503:
            return "🔄 Model is loading. Please wait a moment and try again."
        elif response.status_code == 401:
            return "🔑 Authentication failed. Please check your HF_API_TOKEN."
        elif response.status_code == 402:
            return "💳 This model requires credits. Try a different model or add credits to your HF account."
        elif response.status_code == 404:
            return f"❌ Model '{llm_model}' not found. Try enabling more providers in your HF settings."
            
        response.raise_for_status()
        data = response.json()

        # Parse OpenAI-style response
        if "choices" in data and len(data["choices"]) > 0:
            message = data["choices"][0].get("message", {})
            content = message.get("content", "").strip()
            
            # Add usage info if available (helpful for debugging)
            if "usage" in data:
                usage = data["usage"]
                tokens_used = usage.get("total_tokens", 0)
                if tokens_used > 0:
                    content += f"\n\n*[Used {tokens_used} tokens]*"
            
            return content or "No response generated."
        
        elif "error" in data:
            error_msg = data["error"]
            if isinstance(error_msg, dict):
                return f"API Error: {error_msg.get('message', str(error_msg))}"
            return f"API Error: {error_msg}"
        
        return f"Unexpected response format: {str(data)}"

    except requests.exceptions.Timeout:
        return "⏰ Request timed out. The model might be busy. Try again."
    except requests.exceptions.RequestException as e:
        return f"🌐 Network error: {str(e)}"
    except json.JSONDecodeError:
        return "📄 Invalid JSON response from API."
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"





# ----------------------------
# App Config
# ----------------------------
st.set_page_config(
    page_title="Personal Codex Agent (RAG)",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Personal Codex Agent — RAG-powered")
st.caption("Ubundi Trial Project: context-aware agent that answers questions about you, from your own docs.")


# ----------------------------
# Helper Functions
# ----------------------------
def require_deps():
    if MISSING_DEPS:
        st.error(
            "Missing dependencies: " + ", ".join(MISSING_DEPS) +
            "\n\nPlease install: pip install streamlit chromadb pypdf torch transformers"
        )
        st.stop()


def read_pdf(file: io.BytesIO) -> str:
    if not pypdf:
        return ""
    try:
        reader = pypdf.PdfReader(file)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except Exception:
        return ""


def read_text_like(uploaded) -> str:
    try:
        return uploaded.read().decode("utf-8", errors="ignore")
    except Exception:
        try:
            return uploaded.getvalue().decode("utf-8", errors="ignore")
        except Exception:
            return ""


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if end == n:
            break
    return [c.strip() for c in chunks if c.strip()]


# ----------------------------
# Sidebar: Setup & Docs
# ----------------------------
with st.sidebar:
    st.header("Setup")
    
    st.markdown("---")
    st.subheader("Optional: Hugging Face API")
    hf_token_input = st.text_input("Hugging Face API Token", type="password", help="Add HF API token for faster cloud inference")
    if hf_token_input:
        st.session_state.hf_api_token = hf_token_input

    st.subheader("Local Language Model")
    # Update your model selection with working models:
    llm_model = st.selectbox(
        "HuggingFace Language Model",
        [
            "meta-llama/Llama-3.1-8B-Instruct",        # ✅ Working - excellent for chat
            "meta-llama/Llama-3.1-70B-Instruct",       # ✅ Working - best quality (slower)
            "microsoft/Phi-3-mini-4k-instruct",        # ✅ Working - fast and efficient
            "mistralai/Mistral-7B-Instruct-v0.3",      # ✅ Working - good balance
            "google/gemma-2-9b-it",                     # ✅ Working - Google's model
            "deepseek-ai/deepseek-coder-6.7b-instruct", # ✅ Working - code-focused
            "HuggingFaceH4/zephyr-7b-beta",             # ✅ Working - instruction following
            "microsoft/DialoGPT-medium",                # Fallback option
        ],
        help="All models tested and working with Inference Providers API. Llama 3.1 8B recommended for best balance of speed and quality."  
    )
    st.session_state.llm_model_name = llm_model

    st.markdown("---")
    st.subheader("Persona & Modes")
    default_voice = (
        "You are Edith's personal codex. Speak in Edith's voice: clear, curious, and pragmatic. "
        "Be specific, cite sources from the documents by title where relevant, and avoid generic filler."
    )
    system_voice = st.text_area("Base persona/system prompt", value=default_voice, height=120)

    mode = st.selectbox(
        "Answering mode",
        [
            "Interview mode (concise, professional)",
            "Personal storytelling (reflective, narrative)",
            "Fast facts (bullet points)",
            "Humble brag (confident, truthful)",
            "Self-reflection (strengths, growth, collaboration)",
        ],
    )

    top_k = st.slider("Retriever: top_k", min_value=2, max_value=10, value=5, step=1)
    temperature = st.slider("LLM temperature", min_value=0.0, max_value=1.2, value=0.7, step=0.1)
    max_response_length = st.slider("Max response length", min_value=100, max_value=1000, value=400, step=50)
    st.session_state.temperature = temperature
    st.session_state.max_response_length = max_response_length

    st.markdown("---")
    st.subheader("Embedding Model")
    embedding_model = st.selectbox(
        "HuggingFace Embedding Model",
        [
            "sentence-transformers/all-MiniLM-L6-v2",  # 90MB, fast
            "microsoft/MiniLM-L12-H384-uncased",       # 50MB, faster
            "distilbert-base-uncased",                  # 250MB, good quality
            "bert-base-uncased",                        # 440MB, best quality
        ],
        help="Choose the HuggingFace model for embeddings. Models with 'sentence-transformers/' prefix are optimized for embeddings."
    )

    st.markdown("---")
    st.subheader("Upload & Index Your Docs")
    uploaded_files = st.file_uploader(
        "Upload CV + supporting docs (PDF, TXT, or MD)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Minimum: 1 CV + 2 supporting docs",
    )
    do_rebuild = st.button("🔄 Build / Rebuild Index")


# ----------------------------
# Persistent Paths
# ----------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(APP_DIR, ".chroma")
COLLECTION_NAME = "candidate_codex"
CACHE_DIR = Path(APP_DIR) / ".embedding_cache"
CACHE_DIR.mkdir(exist_ok=True)

if "artifact_log" not in st.session_state:
    st.session_state.artifact_log = []
if "collection_ready" not in st.session_state:
    st.session_state.collection_ready = False
if "history" not in st.session_state:
    st.session_state.history = []


# ----------------------------
# Embedding Cache Functions
# ----------------------------
def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_cached_embedding(text: str, model_name: str):
    key = hashlib.sha256(f"{text}_{model_name}".encode("utf-8")).hexdigest()
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    return None


def save_cached_embedding(text: str, embedding: List[float], model_name: str):
    key = hashlib.sha256(f"{text}_{model_name}".encode("utf-8")).hexdigest()
    cache_file = CACHE_DIR / f"{key}.json"
    cache_file.write_text(json.dumps(embedding))


def embed_text_with_cache(text: str, ef, model_name: str):
    cached = get_cached_embedding(text, model_name)
    if cached:
        return cached
    embedding = ef([text])[0]
    save_cached_embedding(text, embedding, model_name)
    return embedding


def embed_text_with_cache_batch(texts: List[str], ef, model_name: str):
    """Process multiple texts efficiently, using cache when possible"""
    embeddings = []
    uncached_texts = []
    uncached_indices = []
    
    # Check cache first
    for i, text in enumerate(texts):
        cached = get_cached_embedding(text, model_name)
        if cached:
            embeddings.append(cached)
        else:
            embeddings.append(None)  # Placeholder
            uncached_texts.append(text)
            uncached_indices.append(i)
    
    # Process uncached texts in batch
    if uncached_texts:
        batch_embeddings = ef(uncached_texts)
        for idx, emb in zip(uncached_indices, batch_embeddings):
            embeddings[idx] = emb
            save_cached_embedding(texts[idx], emb, model_name)
    
    return embeddings


# ----------------------------
# Build / Rebuild Vector Index
# ----------------------------
def build_index(files, model_name: str) -> Dict[str, Any]:
    require_deps()
    
    # Initialize HuggingFace embedding function
    try:
        ef = HFEmbeddingFunction(model_name=model_name)
    except Exception as e:
        return {"ok": False, "error": f"Failed to load HuggingFace model: {str(e)}"}

    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(is_persistent=True))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    col = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,  # embeddings supplied manually
        metadata={"model": model_name}
    )

    all_chunks, metadatas, ids, all_embeddings = [], [], [], []

    if not files:
        return {"ok": False, "error": "Please upload at least one file."}

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_chunks = 0
    for file_idx, f in enumerate(files):
        status_text.text(f"Processing file {file_idx + 1}/{len(files)}: {f.name}")
        
        name = f.name
        ext = name.split(".")[-1].lower()
        if ext == "pdf":
            text = read_pdf(f)
        else:
            text = read_text_like(f)

        if not text:
            continue

        chunks = chunk_text(text)
        
        # Process chunks in batches for efficiency
        chunk_batch_size = 16  # Process embeddings in batches
        for i in range(0, len(chunks), chunk_batch_size):
            batch_chunks = chunks[i:i + chunk_batch_size]
            
            # Get embeddings for the batch
            batch_embeddings = embed_text_with_cache_batch(batch_chunks, ef, model_name)
            
            for j, (ch, emb) in enumerate(zip(batch_chunks, batch_embeddings)):
                uid = f"{name}::{i+j}::{int(time.time()*1000)}"
                all_chunks.append(ch)
                metadatas.append({"source": name, "chunk": i+j})
                ids.append(uid)
                all_embeddings.append(emb)
                total_chunks += 1
        
        progress_bar.progress((file_idx + 1) / len(files) * 0.8)  # 80% for processing

    if not all_chunks:
        return {"ok": False, "error": "No extractable text found."}

    status_text.text("Adding to vector database...")
    BATCH_SIZE = 50
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch_docs = all_chunks[i:i+BATCH_SIZE]
        batch_metas = metadatas[i:i+BATCH_SIZE]
        batch_ids = ids[i:i+BATCH_SIZE]
        batch_embs = all_embeddings[i:i+BATCH_SIZE]
        col.add(documents=batch_docs, metadatas=batch_metas, ids=batch_ids, embeddings=batch_embs)

    progress_bar.progress(1.0)
    status_text.text("Index building complete!")
    
    return {"ok": True, "count": len(all_chunks)}


if do_rebuild:
    with st.spinner("Building vector index with HuggingFace embeddings..."):
        result = build_index(uploaded_files, embedding_model)
    if result.get("ok"):
        st.success(f"Indexed {result['count']} chunks ✅")
        st.session_state.collection_ready = True
    else:
        st.error(result.get("error", "Unknown error while building index."))
        st.session_state.collection_ready = False


# ----------------------------
# Retrieval + LLM Call
# ----------------------------
def get_collection():
    if not chromadb:
        return None
    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(is_persistent=True))
    try:
        col = client.get_collection(name=COLLECTION_NAME)
        return col
    except Exception:
        return None


def retrieve(query: str, k: int, model_name: str) -> List[Dict[str, Any]]:
    col = get_collection()
    if not col:
        return []
    
    # Reuse cached embedding function
    cache_key = f"hf_model_{model_name}"
    if cache_key not in st.session_state:
        # Initialize if not cached
        try:
            ef = HFEmbeddingFunction(model_name=model_name)
        except Exception:
            return []
    else:
        # Create lightweight wrapper for cached model
        class CachedEF:
            def __init__(self):
                self.tokenizer = st.session_state[cache_key]['tokenizer']
                self.model = st.session_state[cache_key]['model']
                
            def mean_pooling(self, model_output, attention_mask):
                token_embeddings = model_output[0]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
            def __call__(self, texts):
                inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
                with torch.no_grad():
                    model_output = self.model(**inputs)
                embeddings = self.mean_pooling(model_output, inputs['attention_mask'])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                return embeddings.cpu().numpy().tolist()
        
        ef = CachedEF()
    
    try:
        query_embedding = ef([query])[0]
    except Exception:
        return []
    
    out = col.query(
        query_embeddings=[query_embedding], 
        n_results=k, 
        include=["documents", "metadatas", "distances"]
    )
    docs = out.get("documents", [[]])[0]
    metas = out.get("metadatas", [[]])[0]
    dists = out.get("distances", [[]])[0]
    return [{"document": docs[i], "metadata": metas[i], "distance": dists[i]} for i in range(len(docs))]


def mode_instructions(selected: str) -> str:
    if selected.startswith("Interview"):
        return "Answer concisely and professionally (2–5 sentences)."
    if selected.startswith("Personal storytelling"):
        return "Answer in a reflective, narrative tone with specific anecdotes where possible."
    if selected.startswith("Fast facts"):
        return "Answer in short bullet points. Be direct."
    if selected.startswith("Humble brag"):
        return "Answer confidently. Highlight achievements, metrics, and concrete impact."
    if selected.startswith("Self-reflection"):
        return ("Reflect on strengths, areas for growth, collaboration style, and tasks that energize/drain. "
                "Be honest, specific, and grounded in the documents.")
    return "Be helpful and specific."


def build_prompt(user_q: str, retrieved: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    cites = []
    for r in retrieved:
        meta = r.get("metadata", {})
        src = meta.get("source", "")
        idx = meta.get("chunk", 0)
        snippet = r.get("document", "")[:300].replace("\n", " ")
        cites.append(f"- {src} [chunk {idx}]: {snippet}")

    context_block = (
        "You can only use the following retrieved context to answer. If something is unknown, say you don't know.\n\n"
        "Retrieved context (with sources):\n" + "\n".join(cites)
    )

    messages = [
        {"role": "system", "content": system_voice},
        {"role": "system", "content": mode_instructions(mode)},
        {"role": "system", "content": context_block},
        {"role": "user", "content": user_q},
    ]
    return messages


# ----------------------------
# Chat UI
# ----------------------------
st.markdown("""
**How to use**
1) Choose your local language model in the sidebar
2) Upload CV + 2–3 docs and click **Build / Rebuild Index**  
3) Ask questions like *"What kind of engineer are you?"* or *"What projects are you most proud of?"*  
4) Switch **modes** in the sidebar to change tone

""")

# Add model info
if st.session_state.get("llm_model_name"):
    with st.expander("ℹ️ Current Model Info"):
        model = st.session_state.llm_model_name
        if "flan-t5" in model:
            st.info("📚 **T5 Model**: Optimized for question-answering and instruction following. Good for factual responses.")
        elif "dialogpt" in model.lower():
            st.info("💬 **DialoGPT Model**: Designed for conversations. Good for natural dialogue.")
        elif "gpt2" in model.lower():
            st.info("🔤 **GPT-2 Model**: General purpose text generation. Good for creative and varied responses.")
        
        st.caption("**First run**: Model will download (80MB-500MB depending on selection). Subsequent runs are much faster!")

user_q = st.chat_input("Ask about the candidate...")

for role, content in st.session_state.history:
    with st.chat_message(role): st.markdown(content)

if user_q:
    with st.chat_message("user"):
        st.markdown(user_q)
    st.session_state.history.append(("user", user_q))
    st.session_state.artifact_log.append({"ts": datetime.utcnow().isoformat(), "role": "user", "content": user_q})

    # Retrieve
    with st.spinner("Retrieving relevant context..."):
        retrieved = retrieve(user_q, top_k, embedding_model)

    # Pretty-show sources accordion
    with st.expander("🔍 Retrieved sources (top_k)"):
        if not retrieved:
            st.info("No context retrieved. Did you build the index?")
        for r in retrieved:
            meta = r.get("metadata", {})
            st.markdown(f"**{meta.get('source','?')}** – chunk {meta.get('chunk',0)} (distance: {r.get('distance', 0):.3f})")
            st.code((r.get("document", "")[:800]).strip())

    # Build prompt and call local LLM
    messages = build_prompt(user_q, retrieved)
    # Log artifacts (prompt messages)
    for m in messages:
        st.session_state.artifact_log.append({"ts": datetime.utcnow().isoformat(), "role": m["role"], "content": m["content"]})

    with st.spinner("Generating response with local LLM..."):
        answer = call_llm(messages) or "(No answer)"

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.history.append(("assistant", answer))
    st.session_state.artifact_log.append({"ts": datetime.utcnow().isoformat(), "role": "assistant", "content": answer})
