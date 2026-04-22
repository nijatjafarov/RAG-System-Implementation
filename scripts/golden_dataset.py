import os
import re
import json
import time
import random
import hashlib
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional

from tqdm import tqdm
from openai import OpenAI


INPUT_FILE = "../data/main_chunks.json"
OUTPUT_FILE = "../data/golden_qa_dataset.json"
STATS_FILE = "../data/azragbench_stats.json"
FAILED_FILE = "../data/azragbench_failed.json"
COVERAGE_FILE = "../data/azragbench_coverage.json"


API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


MAX_RETRIES = 4
TEMPERATURE = 0.2
TARGET_TOTAL = 1000


TARGET_SPLIT = {
    "single_chunk_answerable": 500,
    "multi_chunk_answerable": 250,
    "insufficient_context_in_corpus": 150,
    "insufficient_context_in_retrieved_context": 100,
}


QUESTION_TYPES = [
    "definition",
    "rule_requirement",
    "rights_obligations",
    "eligibility",
    "procedure",
    "authority_responsibility",
    "timeline_date",
    "event_factual",
    "cause_effect",
    "comparison",
    "exception_condition",
]


QUESTION_TYPE_TARGETS = {
    "definition": 100,
    "rule_requirement": 130,
    "rights_obligations": 90,
    "eligibility": 80,
    "procedure": 100,
    "authority_responsibility": 90,
    "timeline_date": 90,
    "event_factual": 110,
    "cause_effect": 70,
    "comparison": 70,
    "exception_condition": 70,
}


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, list):
        text = " ; ".join(str(x).strip() for x in text if str(x).strip())
    elif isinstance(text, dict):
        text = json.dumps(text, ensure_ascii=False)
    else:
        text = str(text)

    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_answer(answer: Any) -> str:
    if isinstance(answer, list):
        cleaned = [normalize_text(x) for x in answer if normalize_text(x)]
        return "; ".join(cleaned)
    return normalize_text(answer)


def safe_json_value(x: Any, default: str = "") -> str:
    v = normalize_text(x)
    return v if v else default


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def save_json(path: str, rows: List[Dict[str, Any]]):
    """Save as JSON array instead of JSONL"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def append_json(path: str, row: Dict[str, Any]):
    """Append to JSON array - reads entire file, adds item, writes back"""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
            except:
                data = []
    else:
        data = []
    
    data.append(row)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_chunks(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("chunks", [data])
    return data


def clean_text_for_grouping(text: str) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"[^a-zA-ZəöğüşçıƏÖĞÜŞÇİ0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", clean_text_for_grouping(text))


def overlap_score(a: str, b: str) -> float:
    sa = set(tokenize(a))
    sb = set(tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def extract_key_for_group(chunk: Dict[str, Any]) -> str:
    title = normalize_text(chunk.get("title", ""))
    doc_id = normalize_text(chunk.get("doc_id", ""))
    if title:
        return title[:180]
    return doc_id


def is_good_chunk(chunk: Dict[str, Any]) -> bool:
    text = normalize_text(chunk.get("text", ""))
    return len(text) >= 250 and len(text.split()) >= 35


def sample_question_type(counter: Counter) -> str:
    deficits = {
        q: QUESTION_TYPE_TARGETS[q] - counter[q]
        for q in QUESTION_TYPES
    }
    positive = {k: v for k, v in deficits.items() if v > 0}
    if positive:
        return max(positive.items(), key=lambda x: x[1])[0]
    return random.choice(QUESTION_TYPES)


def make_id(prefix: str, question: str, answer: str) -> str:
    h = sha1_text(prefix + "|||" + question + "|||" + answer)[:16]
    return f"azragbench_{h}"


def evidence_from_answer(answer: Any, contexts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out = []
    answer_str = normalize_answer(answer)
    ans = answer_str.lower()
    ans_words = set(re.findall(r"\w+", ans))

    if not ans:
        return out

    for c in contexts:
        txt = normalize_text(c.get("text", ""))
        txt_l = txt.lower()

        if ans and ans in txt_l:
            out.append({"chunk_id": c["chunk_id"], "evidence_text": answer_str[:500]})
            continue

        sents = re.split(r'(?<=[\.\!\?])\s+|(?<=:)\s+', txt)
        best = ""
        best_score = 0
        for sent in sents:
            sw = set(re.findall(r"\w+", sent.lower()))
            score = len(ans_words & sw)
            if score > best_score:
                best_score = score
                best = sent

        if best_score >= max(2, int(len(ans_words) * 0.35)):
            out.append({"chunk_id": c["chunk_id"], "evidence_text": best[:500]})

    return out


def question_fingerprint(question: str) -> str:
    q = normalize_text(question).lower()
    q = re.sub(r"[^\w\s]", "", q)
    return q


def build_related_groups(chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups = defaultdict(list)
    for chunk in chunks:
        key = extract_key_for_group(chunk)
        groups[key].append(chunk)
    return groups


def build_multi_chunk_candidates(chunks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    groups = build_related_groups(chunks)
    candidates = []

    # same-title groups first
    for _, group in groups.items():
        if len(group) >= 2:
            group = sorted(group, key=lambda x: x.get("chunk_id", ""))
            for i in range(len(group) - 1):
                pair = [group[i], group[i + 1]]
                candidates.append(pair)
                if len(group) >= 3 and i + 2 < len(group):
                    candidates.append([group[i], group[i + 1], group[i + 2]])

    # semantic overlap fallback
    random_chunks = chunks[:]
    random.shuffle(random_chunks)
    for i in range(min(300, len(random_chunks))):
        for j in range(i + 1, min(i + 10, len(random_chunks))):
            c1, c2 = random_chunks[i], random_chunks[j]
            if c1["doc_id"] == c2["doc_id"]:
                continue
            if overlap_score(c1.get("title", "") + " " + c1.get("text", "")[:300],
                             c2.get("title", "") + " " + c2.get("text", "")[:300]) > 0.18:
                candidates.append([c1, c2])

    return candidates


SYSTEM_GEN = """
You are building a GOLDEN Azerbaijani RAG benchmark.

IMPORTANT:
- Generate only Azerbaijani questions and answers.
- Never use external world knowledge unless explicitly asked for impossible/unanswerable design.
- Questions must be benchmark-quality, not vague or generic.
- Focus on retrieval usefulness, faithfulness, and answerability.
- "answer" MUST always be a single string.
- If the answer is naturally a list, join items into one semicolon-separated Azerbaijani string.
- Never return answer as JSON array or object.

Return only valid JSON.
"""


def prompt_single(chunk: Dict[str, Any], qtype: str) -> str:
    return f"""
Create 1 high-quality Azerbaijani QA pair from this SINGLE chunk.

Requirements:
- The answer MUST be directly answerable from this chunk alone.
- The question should be useful for legal/news/encyclopedic RAG evaluation.
- Avoid generic summary questions.
- Use this preferred question_type if naturally possible: {qtype}

Return JSON:
{{
  "question": "...",
  "answer": "...",
  "question_type": "{qtype}",
  "difficulty": "easy|medium|hard",
  "answer_style": "short_exact|short_abstractive|list|yes_no",
  "why_grounded": "very short explanation"
}}

CHUNK:
Title: {chunk.get("title")}
Doc ID: {chunk.get("doc_id")}
Chunk ID: {chunk.get("chunk_id")}
Published at: {chunk.get("published_at")}
Text:
\"\"\"
{chunk.get("text", "")}
\"\"\"
"""


def prompt_multi(chunks: List[Dict[str, Any]], qtype: str) -> str:
    joined = "\n\n".join([
        f"[CHUNK {i+1}] chunk_id={c['chunk_id']} title={c.get('title','')}\n{c.get('text','')}"
        for i, c in enumerate(chunks)
    ])
    return f"""
Create 1 high-quality Azerbaijani MULTI-CHUNK QA pair.

Requirements:
- The question MUST require combining information from at least 2 chunks.
- It should NOT be answerable from only one chunk.
- Use this preferred question_type if naturally possible: {qtype}
- Make it realistic and benchmark-worthy.

Return JSON:
{{
  "question": "...",
  "answer": "...",
  "question_type": "{qtype}",
  "difficulty": "easy|medium|hard",
  "answer_style": "short_exact|short_abstractive|list|yes_no",
  "why_multichunk": "very short explanation"
}}

CHUNKS:
{joined}
"""


def prompt_impossible_corpus(context_chunks: List[Dict[str, Any]], qtype: str) -> str:
    joined = "\n\n".join([
        f"[CHUNK {i+1}] chunk_id={c['chunk_id']} title={c.get('title','')}\n{c.get('text','')}"
        for i, c in enumerate(context_chunks)
    ])
    return f"""
Create 1 high-quality Azerbaijani UNANSWERABLE question.

Goal:
- The question must sound realistic and relevant to the topic of the provided chunks.
- BUT the answer must NOT be present in the provided context.
- It should also NOT be trivially inferable.
- The correct system behavior should be to abstain.
- The question should ideally look answerable at first glance, but actually not be answerable from context.

Return JSON:
{{
  "question": "...",
  "ideal_abstention_answer": "Verilmiş məlumatlar əsasında bu suala cavab vermək mümkün deyil.",
  "question_type": "{qtype}",
  "difficulty": "easy|medium|hard",
  "why_unanswerable": "very short explanation"
}}

CONTEXT CHUNKS:
{joined}
"""


def llm_json(prompt: str, model: str = MODEL) -> Optional[Dict[str, Any]]:
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_GEN},
                    {"role": "user", "content": prompt},
                ],
            )
            content = resp.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            wait = (2 ** attempt) + random.random()
            print(f"[WARN] retry {attempt+1}: {e}")
            time.sleep(wait)
    return None


def make_single_sample(chunk: Dict[str, Any], qtype: str) -> Optional[Dict[str, Any]]:
    data = llm_json(prompt_single(chunk, qtype), MODEL)
    if not data:
        return None

    q = safe_json_value(data.get("question", ""))
    a = normalize_answer(data.get("answer", ""))
    qt = safe_json_value(data.get("question_type", qtype), qtype)
    difficulty = safe_json_value(data.get("difficulty", "medium"), "medium")
    answer_style = safe_json_value(data.get("answer_style", "short_exact"), "short_exact")

    if not q or not a:
        return None

    context = [{
        "chunk_id": chunk["chunk_id"],
        "text": chunk["text"]
    }]
    evidence = evidence_from_answer(a, context)

    return {
        "id": make_id("single", q, a),
        "question": q,
        "answer": a,
        "answerable": True,
        "benchmark_type": "single_chunk_answerable",
        "question_type": qt if qt in QUESTION_TYPES else qtype,
        "difficulty": difficulty,
        "language": "az",
        "source_doc_ids": [chunk["doc_id"]],
        "context_chunk_ids": [chunk["chunk_id"]],
        "gold_evidence": evidence,
        "abstention_expected": False,
        "answer_style": answer_style,
        "context": context,
    }


def make_multi_sample(chunks: List[Dict[str, Any]], qtype: str) -> Optional[Dict[str, Any]]:
    data = llm_json(prompt_multi(chunks, qtype), MODEL)
    if not data:
        return None

    q = safe_json_value(data.get("question", ""))
    a = normalize_answer(data.get("answer", ""))
    qt = safe_json_value(data.get("question_type", qtype), qtype)
    difficulty = safe_json_value(data.get("difficulty", "hard"), "hard")
    answer_style = safe_json_value(data.get("answer_style", "short_abstractive"), "short_abstractive")

    if not q or not a:
        return None

    context = [{"chunk_id": c["chunk_id"], "text": c["text"]} for c in chunks]
    evidence = evidence_from_answer(a, context)

    return {
        "id": make_id("multi", q, a),
        "question": q,
        "answer": a,
        "answerable": True,
        "benchmark_type": "multi_chunk_answerable",
        "question_type": qt if qt in QUESTION_TYPES else qtype,
        "difficulty": difficulty,
        "language": "az",
        "source_doc_ids": sorted(list(set(c["doc_id"] for c in chunks))),
        "context_chunk_ids": [c["chunk_id"] for c in chunks],
        "gold_evidence": evidence,
        "abstention_expected": False,
        "answer_style": answer_style,
        "context": context,
    }


def make_impossible_corpus_sample(chunks: List[Dict[str, Any]], qtype: str) -> Optional[Dict[str, Any]]:
    data = llm_json(prompt_impossible_corpus(chunks, qtype), MODEL)
    if not data:
        return None

    q = safe_json_value(data.get("question", ""))
    a = normalize_answer(
        data.get("ideal_abstention_answer", "Verilmiş məlumatlar əsasında bu suala cavab vermək mümkün deyil.")
    )
    qt = safe_json_value(data.get("question_type", qtype), qtype)
    difficulty = safe_json_value(data.get("difficulty", "medium"), "medium")

    if not q:
        return None

    context = [{"chunk_id": c["chunk_id"], "text": c["text"]} for c in chunks]

    return {
        "id": make_id("impossible_corpus", q, a),
        "question": q,
        "answer": a,
        "answerable": False,
        "benchmark_type": "insufficient_context_in_corpus",
        "question_type": qt if qt in QUESTION_TYPES else qtype,
        "difficulty": difficulty,
        "language": "az",
        "source_doc_ids": [],
        "context_chunk_ids": [c["chunk_id"] for c in chunks],
        "gold_evidence": [],
        "abstention_expected": True,
        "answer_style": "abstain",
        "context": context,
    }


def make_insufficient_retrieved_sample(answerable_sample: Dict[str, Any], distractor_chunks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    q = answerable_sample["question"]
    a = "Verilmiş məlumatlar əsasında bu suala cavab vermək mümkün deyil."

    context = [{"chunk_id": c["chunk_id"], "text": c["text"]} for c in distractor_chunks]

    return {
        "id": make_id("insufficient_retrieved", q, a),
        "question": q,
        "answer": a,
        "answerable": False,
        "benchmark_type": "insufficient_context_in_retrieved_context",
        "question_type": answerable_sample["question_type"],
        "difficulty": answerable_sample["difficulty"],
        "language": "az",
        "source_doc_ids": answerable_sample["source_doc_ids"],
        "context_chunk_ids": [c["chunk_id"] for c in distractor_chunks],
        "gold_evidence": [],
        "abstention_expected": True,
        "answer_style": "abstain",
        "context": context,
    }


def find_distractors(target_chunk_ids: List[str], chunks: List[Dict[str, Any]], k: int = 2) -> List[Dict[str, Any]]:
    pool = [c for c in chunks if c["chunk_id"] not in set(target_chunk_ids)]
    random.shuffle(pool)
    return pool[:k]


def update_coverage(sample: Dict[str, Any], coverage: Dict[str, Any]):

    for chunk_id in sample["context_chunk_ids"]:
        coverage["chunks_used_as_context"].add(chunk_id)

    for doc_id in sample["source_doc_ids"]:
        coverage["docs_used"].add(doc_id)
        coverage["doc_question_count"][doc_id] += 1
        coverage["question_type_per_doc"][doc_id].append(sample["question_type"])

    if sample["benchmark_type"] == "insufficient_context_in_retrieved_context":
        for chunk_id in sample["context_chunk_ids"]:
            coverage["chunks_used_as_distractor"].add(chunk_id)


def main():
    chunks = [c for c in load_chunks(INPUT_FILE) if is_good_chunk(c)]
    print(f"Loaded good chunks: {len(chunks)}")

    if len(chunks) < 50:
        print("[WARN] You may not have enough chunks for a good benchmark.")

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    if os.path.exists(FAILED_FILE):
        os.remove(FAILED_FILE)

    results = []
    failed = []
    seen_questions = set()
    qtype_counter = Counter()
    split_counter = Counter()

    coverage = {
        "docs_used": set(),
        "chunks_used_as_gold": set(),
        "chunks_used_as_context": set(),
        "chunks_used_as_distractor": set(),
        "doc_question_count": defaultdict(int),
        "chunk_question_count": defaultdict(int),
        "question_type_per_doc": defaultdict(list),
    }

    multi_candidates = build_multi_chunk_candidates(chunks)
    random.shuffle(chunks)
    random.shuffle(multi_candidates)

    # Single-chunk
    print("Generating single-chunk QA...")
    for chunk in tqdm(chunks):
        if split_counter["single_chunk_answerable"] >= TARGET_SPLIT["single_chunk_answerable"]:
            break

        qtype = sample_question_type(qtype_counter)

        try:
            sample = make_single_sample(chunk, qtype)
        except Exception as e:
            fail_row = {
                "stage": "single_exception",
                "chunk_id": chunk.get("chunk_id"),
                "error": str(e)
            }
            failed.append(fail_row)
            append_json(FAILED_FILE, fail_row)
            continue

        if not sample:
            fail_row = {"stage": "single", "chunk_id": chunk["chunk_id"]}
            failed.append(fail_row)
            append_json(FAILED_FILE, fail_row)
            continue

        qfp = question_fingerprint(sample["question"])
        if qfp in seen_questions:
            continue

        results.append(sample)
        append_json(OUTPUT_FILE, sample)
        seen_questions.add(qfp)
        qtype_counter[sample["question_type"]] += 1
        split_counter[sample["benchmark_type"]] += 1
        update_coverage(sample, coverage)

    # Multi-chunk
    print("Generating multi-chunk QA...")
    for pair in tqdm(multi_candidates):
        if split_counter["multi_chunk_answerable"] >= TARGET_SPLIT["multi_chunk_answerable"]:
            break

        qtype = sample_question_type(qtype_counter)

        try:
            sample = make_multi_sample(pair, qtype)
        except Exception as e:
            fail_row = {
                "stage": "multi_exception",
                "chunk_ids": [c.get("chunk_id") for c in pair],
                "error": str(e)
            }
            failed.append(fail_row)
            append_json(FAILED_FILE, fail_row)
            continue

        if not sample:
            fail_row = {
                "stage": "multi",
                "chunk_ids": [c.get("chunk_id") for c in pair]
            }
            failed.append(fail_row)
            append_json(FAILED_FILE, fail_row)
            continue

        qfp = question_fingerprint(sample["question"])
        if qfp in seen_questions:
            continue

        results.append(sample)
        append_json(OUTPUT_FILE, sample)
        seen_questions.add(qfp)
        qtype_counter[sample["question_type"]] += 1
        split_counter[sample["benchmark_type"]] += 1
        update_coverage(sample, coverage)

    # Insufficient context in corpus
    print("Generating insufficient_context_in_corpus...")
    impossible_context_sets = []
    for i in range(0, len(chunks), 2):
        impossible_context_sets.append(chunks[i:i+2])

    for ctx in tqdm(impossible_context_sets):
        if split_counter["insufficient_context_in_corpus"] >= TARGET_SPLIT["insufficient_context_in_corpus"]:
            break
        if not ctx:
            continue

        qtype = sample_question_type(qtype_counter)

        try:
            sample = make_impossible_corpus_sample(ctx, qtype)
        except Exception as e:
            fail_row = {
                "stage": "impossible_exception",
                "chunk_ids": [c.get("chunk_id") for c in ctx],
                "error": str(e)
            }
            failed.append(fail_row)
            append_json(FAILED_FILE, fail_row)
            continue

        if not sample:
            fail_row = {
                "stage": "impossible",
                "chunk_ids": [c.get("chunk_id") for c in ctx]
            }
            failed.append(fail_row)
            append_json(FAILED_FILE, fail_row)
            continue

        qfp = question_fingerprint(sample["question"])
        if qfp in seen_questions:
            continue

        results.append(sample)
        append_json(OUTPUT_FILE, sample)
        seen_questions.add(qfp)
        qtype_counter[sample["question_type"]] += 1
        split_counter[sample["benchmark_type"]] += 1
        update_coverage(sample, coverage)

    # Insufficient retrieved context
    print("Generating insufficient_context_in_retrieved_context...")
    answerable_pool = [r for r in results if r["answerable"] is True]
    random.shuffle(answerable_pool)

    for s in tqdm(answerable_pool):
        if split_counter["insufficient_context_in_retrieved_context"] >= TARGET_SPLIT["insufficient_context_in_retrieved_context"]:
            break

        qfp = question_fingerprint(sample["question"] + "___insufficient")
        if qfp in seen_questions:
            continue

        results.append(sample)
        append_json(OUTPUT_FILE, sample)
        seen_questions.add(qfp)
        qtype_counter[sample["question_type"]] += 1
        split_counter[sample["benchmark_type"]] += 1
        update_coverage(sample, coverage)

    # Save final results as JSON array
    save_json(OUTPUT_FILE, results[:TARGET_TOTAL])

    stats = {
        "total": len(results[:TARGET_TOTAL]),
        "by_split": dict(Counter(r["benchmark_type"] for r in results[:TARGET_TOTAL])),
        "by_question_type": dict(Counter(r["question_type"] for r in results[:TARGET_TOTAL])),
        "by_difficulty": dict(Counter(r["difficulty"] for r in results[:TARGET_TOTAL])),
        "answerable_vs_not": dict(Counter(r["answerable"] for r in results[:TARGET_TOTAL])),
    }

    coverage_stats = {
        "docs_used_count": len(coverage["docs_used"]),
        "chunks_used_as_gold_count": len(coverage["chunks_used_as_gold"]),
        "chunks_used_as_context_count": len(coverage["chunks_used_as_context"]),
        "chunks_used_as_distractor_count": len(coverage["chunks_used_as_distractor"]),
        "docs_unused_count": len(set(c["doc_id"] for c in chunks) - coverage["docs_used"]),
        "chunks_unused_as_gold_count": len(set(c["chunk_id"] for c in chunks) - coverage["chunks_used_as_gold"]),
        "doc_question_count": dict(coverage["doc_question_count"]),
        "chunk_question_count": dict(coverage["chunk_question_count"]),
        "question_type_per_doc": {k: v for k, v in coverage["question_type_per_doc"].items()},
    }

    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    with open(COVERAGE_FILE, "w", encoding="utf-8") as f:
        json.dump(coverage_stats, f, ensure_ascii=False, indent=2)

    print("\nDone")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(json.dumps({
        "docs_used_count": coverage_stats["docs_used_count"],
        "chunks_used_as_gold_count": coverage_stats["chunks_used_as_gold_count"],
        "chunks_unused_as_gold_count": coverage_stats["chunks_unused_as_gold_count"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()