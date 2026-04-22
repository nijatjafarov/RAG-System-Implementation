from tqdm import tqdm
import anthropic, json, re
from pinecone import Pinecone
import google.genai as genai
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from huggingface_hub import login
import os, json, time, re, numpy as np, torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# API Keys and Configurations
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
KIMI_API_KEY = os.getenv("KIMI_API_KEY")
QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")


if 'HUGGINGFACE_TOKEN' in globals() and HUGGINGFACE_TOKEN:
    login(HUGGINGFACE_TOKEN)

EMBEDDING_CONFIGS = {
    "bge_m3": {
        "display_name": "BGE-M3",
        "index_name": "azrag-bge-m3",
        "type": "local_hf",
        "model_name": "BAAI/bge-m3",
    },
        "snowflake_arctic": {
        "display_name": "Snowflake Arctic",
        "index_name": "azrag-snowflake-arctic",
        "type": "local_hf",
        "model_name": "Snowflake/snowflake-arctic-embed-l-v2.0",
    },
    "qwen3_embedding": {
        "display_name": "Qwen3-Embedding",
        "index_name": "azrag-qwen3-embedding-v4",
        "type": "qwen_api",
        "model_name": "text-embedding-v4",
        "dimensions": 1024,
    },
    "gemini_embedding": {
        "display_name": "Gemini Embedding",
        "index_name": "azrag-gemini-embed2",
        "type": "gemini_api",
        "model_name": "gemini-embedding-001",
    }
}

LLM_CONFIGS = {
    "claude": {"display_name": "Claude Sonnet", "type": "anthropic", "model": "claude-sonnet-4-6"},
    "gpt": {"display_name": "GPT-5.4", "type": "openai", "model": "gpt-5.4"},
    "gemini": {"display_name": "Gemini 3 Flash", "type": "gemini", "model": "gemini-3-flash-preview"},
    "kimi": {
        "display_name": "Kimi K2.5",
        "type": "openai_compat",
        "model": "kimi-k2.5",
        "base_url": "https://api.moonshot.ai/v1",
        "api_key_env": "KIMI_API_KEY",
        "temperature": 1,
    },
    "deepseek": {
        "display_name": "DeepSeek Chat",
        "type": "openai_compat",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
}

_print_lock = Lock()
_file_lock = Lock()


RAG_SYSTEM = """Answer ONLY based on the provided context.
If the answer is not in the context, reply: "Verilmiş məlumatlar əsasında bu suala cavab vermək mümkün deyil."
Be concise. Answer in Azerbaijani."""

def truncate_text(text, max_chars=4000):
    if not text:
        return ""
    return text[:max_chars] + "..." if len(text) > max_chars else text

def safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)

def safe_write_json(path, data):
    with _file_lock:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

def retry_call(fn, *args, retries=3, sleep=2, timeout=60, **kwargs):
    for attempt in range(retries):
        try:
            kwargs_with_timeout = {**kwargs, "timeout": timeout}
            return fn(*args, **kwargs_with_timeout)
        except TypeError:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if attempt == retries - 1:
                    raise
                time.sleep(sleep * (attempt + 1))
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(sleep * (attempt + 1))

def load_benchmark(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items = []
    for r in raw:
        if r.get("difficulty") in ("medium", "hard"):
            items.append({
                "id": r["id"],
                "question": r["question"],
                "answer": r.get("answer", ""),
                "answerable": r.get("answerable", True),
                "difficulty": r.get("difficulty"),
                "abstention_expected": r.get("abstention_expected", not r.get("answerable", True)),
                "context_chunk_ids": r.get("context_chunk_ids", []),
                "context": r.get("context", []),
            })
    return items

def is_abstention(text):
    if not text:
        return False
    low = text.lower()
    return "verilmiş məlumatlar əsasında bu suala cavab vermək mümkün deyil" in low


class QwenEmbedder:
    def __init__(self, api_key, model_name, dimensions=1024):
        self.client = OpenAI(api_key=api_key, base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        self.model = model_name
        self.dimensions = dimensions

    def embed(self, text):
        resp = retry_call(self.client.embeddings.create, model=self.model, input=[text], dimensions=self.dimensions)
        return resp.data[0].embedding

class GeminiEmbedder:
    def __init__(self, api_key, model_name):
        self.client = genai.Client(api_key=api_key)
        self.model = model_name

    def embed(self, text):
        result = retry_call(
            self.client.models.embed_content,
            model=self.model,
            contents=[text],
            config=genai.types.EmbedContentConfig(task_type="retrieval_query"),
        )
        return result.embeddings[0].values

class HFEmbedder:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")

    def embed(self, text):
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

def build_embedder(key):
    cfg = EMBEDDING_CONFIGS[key]
    if cfg["type"] == "qwen_api":
        return QwenEmbedder(QWEN_API_KEY, cfg["model_name"], cfg.get("dimensions", 1024))
    if cfg["type"] == "gemini_api":
        return GeminiEmbedder(GEMINI_API_KEY, cfg["model_name"])
    if cfg["type"] == "local_hf":
        return HFEmbedder(cfg["model_name"])
    raise ValueError(f"Unknown embedder: {cfg['type']}")

class PineconeRetriever:
    def __init__(self, index_name, namespace="main"):
        pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = pc.Index(index_name)
        self.namespace = namespace

    def retrieve(self, query_vec, top_k):
        resp = retry_call(self.index.query, vector=query_vec, top_k=top_k, include_metadata=True, namespace=self.namespace)
        return [{"id": m.id, "score": m.score, "metadata": m.metadata} for m in resp.matches]

class OpenAILLM:
    def __init__(self, model, api_key, base_url=None, temperature=0.0):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    def complete(self, system, user):
        resp = retry_call(self.client.chat.completions.create, model=self.model,
                          messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                          temperature=self.temperature, max_completion_tokens=1024)
        return resp.choices[0].message.content.strip()

class AnthropicLLM:
    def __init__(self, model):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = model

    def complete(self, system, user):
        resp = retry_call(self.client.messages.create, model=self.model, max_tokens=1024,
                          system=system, messages=[{"role": "user", "content": user}], temperature=0.0)
        return resp.content[0].text.strip()

class GeminiLLM:
    def __init__(self, model):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = model

    def complete(self, system, user):
        prompt = f"{system}\n\n{user}"
        resp = retry_call(self.client.models.generate_content, model=self.model, contents=prompt)
        return resp.text.strip()

def build_llm(key):
    cfg = LLM_CONFIGS[key]
    t = cfg["type"]
    if t == "openai":
        return OpenAILLM(cfg["model"], OPENAI_API_KEY)
    if t == "anthropic":
        return AnthropicLLM(cfg["model"])
    if t == "gemini":
        return GeminiLLM(cfg["model"])
    if t == "openai_compat":
        api_key = globals().get(cfg["api_key_env"]) or os.getenv(cfg["api_key_env"], "")
        return OpenAILLM(cfg["model"], api_key, base_url=cfg.get("base_url"), temperature=cfg.get("temperature", 0.0))
    raise ValueError(f"Unknown LLM type: {t}")

def build_rag_prompt(question, passages):
    ctx_parts = []
    for i, p in enumerate(passages, 1):
        text = truncate_text(p.get("metadata", {}).get("text", p.get("metadata", {}).get("embedding_text", "")))
        ctx_parts.append(f"[Passage {i}]\n{text}")
    ctx_block = "\n\n".join(ctx_parts)
    return f"Context:\n{ctx_block}\n\nQuestion: {question}\n\nAnswer:"

def recall_at_k(retrieved, relevant, k):
    if not relevant: return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)

def precision_at_k(retrieved, relevant, k):
    if k == 0: return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / k

def ndcg_at_k(retrieved, relevant, k):
    relevant_set = set(relevant)
    gains = [1.0 if rid in relevant_set else 0.0 for rid in retrieved[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted([1.0]*min(len(relevant), k) + [0.0]*max(0, k-len(relevant)), reverse=True)
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0

def mrr_score(retrieved, relevant):
    relevant_set = set(relevant)
    for i, rid in enumerate(retrieved, 1):
        if rid in relevant_set:
            return 1.0 / i
    return 0.0

# Cache for semantic similarity model
_OPENAI_SIM_CLIENT = None

def semantic_similarity(pred, gold, model="text-embedding-3-small", dimensions=1536):
    if not pred or not gold:
        return 0.0

    global _OPENAI_SIM_CLIENT

    if _OPENAI_SIM_CLIENT is None:
        _OPENAI_SIM_CLIENT = OpenAI(api_key=OPENAI_API_KEY)

    resp = retry_call(
        _OPENAI_SIM_CLIENT.embeddings.create,
        model=model,
        input=[pred, gold],
        dimensions=dimensions,
        timeout=30
    )
    
    vecs = [item.embedding for item in resp.data]
    if len(vecs) != 2:
        return 0.0
        
    sim = np.clip(np.dot(vecs[0], vecs[1]), 0.0, 1.0)
    return float(sim)

def llm_judge_claude(question, gold_answer, generated, context, answerable):
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        user_msg = (f"""Question: {question}
            Gold Answer: {gold_answer}
            Answerable: {answerable}
            Context: {truncate_text(context, 2000)}
            Generated: {generated}

            Return JSON: {{"score": 0-10, "reason": "..."}}""")
        resp = retry_call(client.messages.create, model="claude-sonnet-4-6", max_tokens=512,
                          system="Score 0-10 on correctness, faithfulness, completeness. Return ONLY AND ONLY JSON. Be objective and strict.",
                          messages=[{"role": "user", "content": user_msg}])
        text_block = next((b for b in resp.content if b.type == "text"), None)
        if not text_block:
            return 0.0, "no text block in response"
        raw = re.sub(r"```json|```", "", text_block.text.strip()).strip()
        if not raw:
            return 0.0, "empty response"
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            return 0.0, f"no JSON found in: {raw[:100]}"
        data = json.loads(match.group())
        return float(data["score"]) / 10.0, data.get("reason", "")
    except Exception as e:
        print(f"Judge error: {e}")
        return 0.0, str(e)

    
def process_retrieval_item(item, embedder, retriever, k):
    try:
        qvec = embedder.embed(item["question"])
        passages = retriever.retrieve(qvec, top_k=k)
        retrieved_ids = [p["id"] for p in passages]
        relevant_ids = item["context_chunk_ids"]
        return {
            "id": item["id"], "success": True, "passages": passages,
            "retrieval_metrics": {
                "question_id": item["id"], "retrieved_ids": retrieved_ids,
                "relevant_ids": relevant_ids, "answerable": item["answerable"],
                "mrr": mrr_score(retrieved_ids, relevant_ids),
                "recall_at_k": {k: recall_at_k(retrieved_ids, relevant_ids, k)},
                "precision_at_k": {k: precision_at_k(retrieved_ids, relevant_ids, k)},
                "ndcg_at_k": {k: ndcg_at_k(retrieved_ids, relevant_ids, k)},
            }
        }
    except Exception as e:
        return {"id": item["id"], "success": False, "error": str(e), "passages": []}

def process_generation_item(item, passages, llm, system_prompt, build_prompt_fn, judge_fn):
    try:
        prompt = build_prompt_fn(item["question"], passages)
        generated = llm.complete(system_prompt, prompt)
        abstained = is_abstention(generated)
        abstention_correct = None
        if item["abstention_expected"]:
            abstention_correct = abstained
        elif abstained:
            abstention_correct = False
        context_texts = [truncate_text(p.get("metadata", {}).get("text", p.get("metadata", {}).get("embedding_text", ""))) for p in passages]
        context_text = " ".join(context_texts)
        judge_score, judge_reason = judge_fn(
            item["question"], item["answer"], generated, context_text, item["answerable"]
        )
        return {
            "question_id": item["id"], "question": item["question"],
            "generated_answer": generated, "gold_answer": item["answer"],
            "answerable": item["answerable"], "abstention_expected": item["abstention_expected"],
            "abstention_correct": abstention_correct,
            "semantic_similarity": semantic_similarity(generated, item["answer"]),
            "llm_judge_score": judge_score, "llm_judge_reason": judge_reason,
            "success": True
        }
    except Exception as e:
        return {
            "question_id": item["id"], "generated_answer": "",
            "semantic_similarity": 0.0, "llm_judge_score": 0.0,
            "llm_judge_reason": str(e), "context_relevance": None, "faithfulness": None, "success": False
        }

def run_pipeline(benchmark_path, output_path, k, embedding_keys, llm_keys, namespace="main",
                 retrieval_workers=8, generation_workers=4, detailed_output_path=None):
    safe_print(f"\nAzRAGBench Evaluation | Benchmark: {benchmark_path} | Output: {output_path}")
    items = load_benchmark(benchmark_path)
    safe_print(f"Loaded {len(items)} items")
    all_results = []
    flat_records = []
    results_lock = Lock()

    # Detailed per-question structure
    detailed_by_qid = {}
    for item in items:
        detailed_by_qid[item["id"]] = {
            "question_id": item["id"],
            "question": item["question"],
            "gold_answer": item["answer"],
            "answerable": item["answerable"],
            "difficulty": item.get("difficulty"),
            "abstention_expected": item["abstention_expected"],
            "context_chunk_ids": item["context_chunk_ids"],
            "combinations": {},
        }

    for emb_key in embedding_keys:
        emb_cfg = EMBEDDING_CONFIGS.get(emb_key)
        if not emb_cfg: continue
        safe_print(f"\n[EMBED] {emb_cfg['display_name']}")
        try:
            embedder = build_embedder(emb_key)
            retriever = PineconeRetriever(emb_cfg["index_name"], namespace)
        except Exception as e:
            safe_print(f"  Setup failed: {e}")
            continue

        safe_print("Retrieving...")
        retrieval_cache = {}
        retrieval_results = {}

        with ThreadPoolExecutor(max_workers=retrieval_workers) as executor:
            future_to_id = {
                executor.submit(process_retrieval_item, item, embedder, retriever, k): item["id"]
                for item in items
            }
            for future in tqdm(as_completed(future_to_id), total=len(items), desc="  Retrieve"):
                item_id = future_to_id[future]
                try:
                    result = future.result()
                    if result["success"]:
                        retrieval_cache[item_id] = result["passages"]
                        retrieval_results[item_id] = result["retrieval_metrics"]
                    else:
                        tqdm.write(f"  Retrieval failed [{item_id}]: {result.get('error', 'unknown')}")
                except Exception as e:
                    safe_print(f"Retrieval error for {item_id}: {e}")

        for llm_key in llm_keys:
            llm_cfg = LLM_CONFIGS.get(llm_key)
            if not llm_cfg: continue
            safe_print(f"[LLM] {llm_cfg['display_name']}")
            try:
                llm = build_llm(llm_key)
            except Exception as e:
                safe_print(f"LLM setup failed: {e}")
                continue

            gen_results = []
            gen_results_by_qid = {}
            safe_print("Generating answers...")

            with ThreadPoolExecutor(max_workers=generation_workers) as executor:
                future_to_item = {
                    executor.submit(
                        process_generation_item,
                        item,
                        retrieval_cache.get(item["id"], []),
                        llm,
                        RAG_SYSTEM,
                        build_rag_prompt,
                        llm_judge_claude
                    ): item
                    for item in items
                }
                for future in tqdm(as_completed(future_to_item), total=len(items), desc="  Generate"):
                    orig_item = future_to_item[future]
                    try:
                        result = future.result()
                        if result["success"]:
                            gen_results.append(result)
                            gen_results_by_qid[orig_item["id"]] = result
                        else:
                            tqdm.write(f"Generation failed [{orig_item['id']}]: {result.get('llm_judge_reason', 'unknown')}")
                    except Exception as e:
                        safe_print(f"Generation error: {e}")
                    time.sleep(0.05)

            def mean(vals):
                vals = [v for v in vals if v is not None]
                return np.mean(vals) if vals else None

            ret_agg = {
                "mrr": mean([r["mrr"] for r in retrieval_results.values()]),
                **{f"recall@{k}": mean([r["recall_at_k"].get(k, 0) for r in retrieval_results.values()])},
                **{f"precision@{k}": mean([r["precision_at_k"].get(k, 0) for r in retrieval_results.values()])},
                **{f"ndcg@{k}": mean([r["ndcg_at_k"].get(k, 0) for r in retrieval_results.values()])},
            }

            gen_agg = {
                "semantic_similarity": mean([g["semantic_similarity"] for g in gen_results]),
                "llm_judge_score": mean([g["llm_judge_score"] for g in gen_results])
            }
            unans = [g for g in gen_results if g["abstention_expected"]]
            if unans:
                gen_agg["abstention_accuracy"] = sum(1 for g in unans if g["abstention_correct"]) / len(unans)

            def fmt(v):
                return f"{v:.3f}" if v is not None else "N/A"

            safe_print(f"\nResults: {emb_cfg['display_name']} × {llm_cfg['display_name']}")
            safe_print(f"Retrieved: {len(retrieval_results)}/{len(items)} | Generated: {len(gen_results)}/{len(items)}")
            safe_print(f"Retrieval: MRR={fmt(ret_agg['mrr'])} | nDCG@{k}={fmt(ret_agg[f'ndcg@{k}'])}")
            safe_print(f"Generation: SemSim={fmt(gen_agg['semantic_similarity'])} | Judge={fmt(gen_agg['llm_judge_score'])}")

            combo_key = f"{emb_key}+{llm_key}"
            combo_result = {
                "embedding_model": emb_key,
                "embedding_display": emb_cfg["display_name"],
                "llm_model": llm_key,
                "llm_display": llm_cfg["display_name"],
                "retrieval_agg": ret_agg,
                "generation_agg": gen_agg,
                "retrieval_per_q": list(retrieval_results.values()),
                "generation_per_q": gen_results,
            }

            # Build detailed per-question entries for this combo
            for item in items:
                qid = item["id"]
                ret_m = retrieval_results.get(qid, {})
                passages = retrieval_cache.get(qid, [])
                gen_r = gen_results_by_qid.get(qid, {})

                flat_records.append({
                    "embedding_model": emb_key,
                    "embedding_display": emb_cfg["display_name"],
                    "llm_model": llm_key,
                    "llm_display": llm_cfg["display_name"],
                    "question_id": qid,
                    "question": item["question"],
                    "gold_answer": item["answer"],
                    "answerable": item["answerable"],
                    "difficulty": item.get("difficulty"),
                    "abstention_expected": item["abstention_expected"],
                    "retrieved_ids": ret_m.get("retrieved_ids", []),
                    "relevant_ids": ret_m.get("relevant_ids", []),
                    "mrr": ret_m.get("mrr"),
                    f"recall@{k}": ret_m.get("recall_at_k", {}).get(k),
                    f"precision@{k}": ret_m.get("precision_at_k", {}).get(k),
                    f"ndcg@{k}": ret_m.get("ndcg_at_k", {}).get(k),
                    "generated_answer": gen_r.get("generated_answer", ""),
                    "abstained": is_abstention(gen_r.get("generated_answer", "")),
                    "abstention_correct": gen_r.get("abstention_correct"),
                    "semantic_similarity": gen_r.get("semantic_similarity"),
                    "llm_judge_score": gen_r.get("llm_judge_score"),
                    "llm_judge_reason": gen_r.get("llm_judge_reason", ""),
                })

                detailed_by_qid[qid]["combinations"][combo_key] = {
                    "embedding_model": emb_key,
                    "embedding_display": emb_cfg["display_name"],
                    "llm_model": llm_key,
                    "llm_display": llm_cfg["display_name"],
                    "retrieval": {
                        "retrieved_ids": ret_m.get("retrieved_ids", []),
                        "relevant_ids": ret_m.get("relevant_ids", []),
                        "passages": [
                            {
                                "id": p["id"],
                                "score": p["score"],
                                "text": truncate_text(p.get("metadata", {}).get("text", p.get("metadata", {}).get("embedding_text", "")), 1000),
                            }
                            for p in passages
                        ],
                        "mrr": ret_m.get("mrr"),
                        f"recall@{k}": ret_m.get("recall_at_k", {}).get(k),
                        f"precision@{k}": ret_m.get("precision_at_k", {}).get(k),
                        f"ndcg@{k}": ret_m.get("ndcg_at_k", {}).get(k),
                    },
                    "generation": {
                        "generated_answer": gen_r.get("generated_answer", ""),
                        "abstained": is_abstention(gen_r.get("generated_answer", "")),
                        "abstention_correct": gen_r.get("abstention_correct"),
                    },
                    "evaluation": {
                        "semantic_similarity": gen_r.get("semantic_similarity"),
                        "llm_judge_score": gen_r.get("llm_judge_score"),
                        "llm_judge_reason": gen_r.get("llm_judge_reason", ""),
                    },
                }

            with results_lock:
                all_results.append(combo_result)
                safe_write_json(output_path, all_results)
                safe_print(f"[SAVED] {output_path}")

                if detailed_output_path:
                    safe_write_json(detailed_output_path, list(detailed_by_qid.values()))
                    safe_print(f"[SAVED DETAILED] {detailed_output_path}")

                    flat_path = detailed_output_path.replace("_detailed", "_flat")
                    safe_write_json(flat_path, flat_records)
                    safe_print(f"[SAVED FLAT] {flat_path}")

    safe_print(f"\n{'='*60}\nLEADERBOARD (sorted by nDCG@{k} × Judge Score)\n{'='*60}")
    def sort_key(r):
        return (r["retrieval_agg"].get(f"ndcg@{k}", 0) or 0) * (r["generation_agg"].get("llm_judge_score", 0) or 0)

    for rank, res in enumerate(sorted(all_results, key=sort_key, reverse=True), 1):
        crel = res['generation_agg'].get('context_relevance', 0) or 0
        faith = res['generation_agg'].get('faithfulness', 0) or 0
        safe_print(f"#{rank:2d} {res['embedding_display']:30s} × {res['llm_display']:25s} | "
                  f"nDCG@{k}={res['retrieval_agg'].get(f'ndcg@{k}',0):.3f} | "
                  f"Judge={res['generation_agg'].get('llm_judge_score',0):.3f} | "
                  f"CR={crel:.3f} | Faith={faith:.3f}")
    safe_print(f"\nFull results: {output_path}")
    if detailed_output_path:
        safe_print(f"Detailed results: {detailed_output_path}")

if __name__ == "__main__":
    run_pipeline(
        "../data/golden_dataset.json",
        "../data/eval_results.json", 5,
        list(EMBEDDING_CONFIGS.keys()),
        [llm for llm in LLM_CONFIGS.keys() if llm != "claude"], "main",
        retrieval_workers=4, generation_workers=4,
        detailed_output_path="../data/eval_results_detailed.json"
    )