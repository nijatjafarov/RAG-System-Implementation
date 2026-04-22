import os
import json
import time
import hashlib
from typing import List, Dict, Any, Iterator
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from google import genai
import torch

load_dotenv()

# API Keys
QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

CHUNKS_PATH = "../data/main_chunks.json"
DEFAULT_NAMESPACE = "main"

# Batch sizes
QWEN_EMBED_BATCH_SIZE = 10
LOCAL_EMBED_BATCH_SIZE = 8
UPSERT_BATCH_SIZE = 100

MAX_RETRIES = 5
RETRY_SLEEP = 3

TRUNCATE_TEXT_METADATA = True
MAX_METADATA_TEXT_LEN = 40000

if HF_TOKEN:
    login(HF_TOKEN)

MODEL_CONFIGS = {
    "qwen3_embedding": {
        "display_name": "Alibaba Qwen3-Embedding (text-embedding-v4)",
        "index_name": "azrag-qwen3-embedding-v4",
        "type": "qwen_api",
        "model_name": "text-embedding-v4",
        "metric": "cosine",
        "dimensions": 1024,
        "max_batch": 10,
    },
    "bge_m3": {
        "display_name": "BAAI BGE-M3",
        "index_name": "azrag-bge-m3",
        "type": "local_hf",
        "model_name": "BAAI/bge-m3",
        "metric": "cosine",
        "trust_remote_code": True,
    },
    "snowflake_arctic": {
        "display_name": "Snowflake Arctic-Embed-L-v2.0",
        "index_name": "azrag-snowflake-arctic",
        "type": "local_hf",
        "model_name": "Snowflake/snowflake-arctic-embed-l-v2.0",
        "metric": "cosine",
        "trust_remote_code": True,
    },
    "gemini_embedding": {
        "display_name": "Google Gemini Embedding 2",
        "index_name": "azrag-gemini-embed2",
        "type": "gemini_api",
        "model_name": "gemini-embedding-001",
        "metric": "cosine",
    }
}

# Utility Functions
def load_chunks(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "chunks" in data:
        return data["chunks"]
    raise ValueError("Unsupported JSON structure. Expected list or {'chunks': [...]}")

def chunked(lst: List[Any], size: int) -> Iterator[List[Any]]:
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

def safe_text(x: Any) -> str:
    return str(x) if x is not None else ""

def build_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    metadata = {
        "chunk_id": safe_text(chunk.get("chunk_id")),
        "doc_id": safe_text(chunk.get("doc_id")),
        "title": safe_text(chunk.get("title")),
        "domain": safe_text(chunk.get("domain")),
        "source": safe_text(chunk.get("source")),
        "url": safe_text(chunk.get("url")),
        "published_at": safe_text(chunk.get("published_at")),
        "corpus_tier": safe_text(chunk.get("corpus_tier")),
        "length": int(chunk.get("length", 0)) if chunk.get("length") is not None else 0,
        "text": safe_text(chunk.get("text")),
        "embedding_text": safe_text(chunk.get("embedding_text")),
    }
    if TRUNCATE_TEXT_METADATA:
        for k, v in metadata.items():
            if isinstance(v, str) and len(v) > MAX_METADATA_TEXT_LEN:
                metadata[k] = v[:MAX_METADATA_TEXT_LEN]
    return metadata

def get_embedding_input(chunk: Dict[str, Any]) -> str:
    return safe_text(chunk.get("embedding_text") or chunk.get("text") or "")

def deterministic_fallback_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

# Embedder Classes
class BaseEmbedder:
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError
    def get_dimension(self) -> int:
        return len(self.embed_texts(["Test"])[0])

class QwenAPIEmbedder(BaseEmbedder):
    def __init__(self, api_key: str, model_name: str, dimensions: int = 1024):
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY is missing.")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        self.model_name = model_name
        self.dimensions = dimensions
        
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=texts,
                    dimensions=self.dimensions,
                    encoding_format="float"
                )
                embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
                return embeddings
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise e
                time.sleep(RETRY_SLEEP * (attempt + 1))
        return []

class GeminiAPIEmbedder(BaseEmbedder):
    def __init__(self, api_key: str, model_name: str):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        for text in texts:
            for attempt in range(MAX_RETRIES):
                try:
                    result = self.model(
                        model=self.model_name,
                        content=text,
                        task_type="retrieval_document"
                    )
                    all_embeddings.append(result['embedding'])
                    break
                except Exception as e:
                    if attempt == MAX_RETRIES - 1:
                        raise e
                    time.sleep(RETRY_SLEEP * (attempt + 1))
        return all_embeddings

class LocalHFEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, trust_remote_code: bool = True):
        print(f"[INFO] Loading local HF model: {model_name}")
        self.model = SentenceTransformer(
            model_name, 
            trust_remote_code=trust_remote_code,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=LOCAL_EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings.tolist()

def create_embedder(model_cfg: Dict[str, Any]) -> BaseEmbedder:
    if model_cfg["type"] == "qwen_api":
        return QwenAPIEmbedder(QWEN_API_KEY, model_cfg["model_name"], model_cfg.get("dimensions", 1024))
    elif model_cfg["type"] == "gemini_api":
        return GeminiAPIEmbedder(GEMINI_API_KEY, model_cfg["model_name"])
    elif model_cfg["type"] == "local_hf":
        return LocalHFEmbedder(model_cfg["model_name"], model_cfg.get("trust_remote_code", True))
    raise ValueError(f"Unknown embedder type: {model_cfg['type']}")

# Vector DB Functions
def ensure_index(pc: Pinecone, index_name: str, dimension: int, metric: str = "cosine"):
    if index_name not in pc.list_indexes().names():
        print(f"[INFO] Creating Pinecone index: {index_name} (dim={dimension}, metric={metric})")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(2)
    else:
        print(f"[INFO] Pinecone index already exists: {index_name}")

def index_model_to_pinecone(chunks: List[Dict[str, Any]], model_key: str, namespace: str = DEFAULT_NAMESPACE):
    model_cfg = MODEL_CONFIGS[model_key]
    print(f"\n{'='*80}\n[START] {model_cfg['display_name']}\n{'='*80}")
    
    embedder = create_embedder(model_cfg)
    
    if model_key == "qwen3_embedding":
        dimension = model_cfg["dimensions"]
    else:
        dimension = embedder.get_dimension()
        
    print(f"[INFO] Embedding dimension: {dimension}")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    ensure_index(pc, model_cfg["index_name"], dimension, model_cfg["metric"])
    index = pc.Index(model_cfg["index_name"])
    
    # Determine batch size for this model
    batch_size = model_cfg.get("max_batch", LOCAL_EMBED_BATCH_SIZE)
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for batch in tqdm(chunked(chunks, batch_size), total=total_batches, desc=f"Embedding {model_key}"):
        texts = [get_embedding_input(c) for c in batch]
        try:
            vectors = embedder.embed_texts(texts)
        except Exception as e:
            print(f"[ERROR] Failed to embed batch: {e}")
            continue
        
        pinecone_records = [
            {
                "id": safe_text(chunk.get("chunk_id")) or deterministic_fallback_id(get_embedding_input(chunk)),
                "values": vector,
                "metadata": build_metadata(chunk)
            }
            for chunk, vector in zip(batch, vectors)
        ]
        
        for attempt in range(MAX_RETRIES):
            try:
                index.upsert(vectors=pinecone_records, namespace=namespace)
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"[ERROR] Upsert failed for batch: {e}")
                time.sleep(RETRY_SLEEP * (attempt + 1))

def main():
    if not PINECONE_API_KEY:
        raise ValueError("Please set PINECONE_API_KEY")
    if not QWEN_API_KEY:
        print("[WARNING] DASHSCOPE_API_KEY not set. Skipping Qwen3.")
    if not GEMINI_API_KEY:
        print("[WARNING] GEMINI_API_KEY not set. Skipping Gemini.")
    
    chunks = load_chunks(CHUNKS_PATH)
    print(f"[INFO] Loaded {len(chunks)} chunks")
    
    # Deduplicate
    seen = set()
    deduped = []
    for c in chunks:
        cid = safe_text(c.get("chunk_id")) or deterministic_fallback_id(get_embedding_input(c))
        c["chunk_id"] = cid
        if cid not in seen:
            seen.add(cid)
            deduped.append(c)
    print(f"[INFO] Deduplicated chunks: {len(deduped)}")
    
    # Process exactly the requested models
    models_to_process = ["qwen3_embedding", "bge_m3", "snowflake_arctic", "gemini_embedding"]
    
    for model_key in models_to_process:
        if model_key == "qwen3_embedding" and not QWEN_API_KEY:
            continue
        if model_key == "gemini_embedding" and not GEMINI_API_KEY:
            continue
            
        try:
            index_model_to_pinecone(deduped, model_key)
        except Exception as e:
            print(f"[ERROR] Critical failure indexing model {model_key}: {e}")
    
    print("\nAll tasks complete.")

if __name__ == "__main__":
    main()