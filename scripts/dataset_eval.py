import json
import os
import time
from typing import Dict, Any, List
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

write_lock = Lock()


def rag_filter_prompt(doc: Dict[str, Any]) -> str:
    text = doc.get("text", doc.get("content", ""))[:2000]

    prompt = f"""
            Evaluate the following Azerbaijani document for inclusion in a high-quality Azerbaijani RAG benchmark.

            IMPORTANT:
            - Be strict.
            - Do NOT inflate ratings.
            - Use the FULL 0–5 range naturally.
            - Weak, shallow, noisy, teaser-like, repetitive, or context-dependent documents should receive LOW values.
            - Only give high values if the document clearly deserves them.
            - Give output in English.

            Document metadata:
            doc_id: {doc.get('doc_id', '')}
            title: {doc.get('title', '')}
            source: {doc.get('source', '')}
            category: {doc.get('category', '')}
            published_at: {doc.get('published_at', '')}

            Document (first 2000 chars):
            \"\"\"{text}\"\"\"

            Return STRICT JSON only:

            {{
            "document_type": "news" | "official" | "educational" | "analysis" | "interview" | "opinion" | "blog" | "reference" | "other",
            "valuable_for_rag": true or false,
            "reasoning_summary": "short reason",
            "strengths": ["..."],
            "weaknesses": ["..."],
            "signals": {{
                "fact_rich": 0-5,
                "self_contained": 0-5,
                "qa_friendly": 0-5,
                "retrieval_friendly": 0-5,
                "informational_depth": 0-5,
                "noise_level": 0-5,
                "context_dependency": 0-5,
                "duplication_or_redundancy": 0-5
            }},
            "questionability": {{
                "estimated_natural_questions": 0-10,
                "likely_question_types": ["who", "what", "when", "where", "why", "how", "definition", "comparison", "policy", "numeric", "procedural"]
            }},
            "notes_for_chunking": "short note"
            }}
        """
    return prompt


def parse_llm_response(response_text: str) -> Dict[str, Any]:
    clean_text = response_text.strip()

    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    elif clean_text.startswith("```"):
        clean_text = clean_text[3:]

    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]

    return json.loads(clean_text.strip())


def compute_rag_score(signals: Dict[str, int]) -> int:
    fact_rich = signals.get("fact_rich", 0)
    self_contained = signals.get("self_contained", 0)
    qa_friendly = signals.get("qa_friendly", 0)
    retrieval_friendly = signals.get("retrieval_friendly", 0)
    informational_depth = signals.get("informational_depth", 0)
    noise_level = signals.get("noise_level", 5)
    context_dependency = signals.get("context_dependency", 5)
    duplication_or_redundancy = signals.get("duplication_or_redundancy", 5)

    positive = (
        fact_rich * 5 +
        self_contained * 4 +
        qa_friendly * 5 +
        retrieval_friendly * 4 +
        informational_depth * 4
    )

    negative = (
        noise_level * 4 +
        context_dependency * 3 +
        duplication_or_redundancy * 3
    )

    raw = positive - negative

    score = int((raw + 35) / 82 * 100)
    score = max(0, min(100, score))
    return score


def derive_keep_priority(score: int) -> str:
    if score >= 85:
        return "high"
    elif score >= 75:
        return "medium"
    elif score >= 65:
        return "low"
    else:
        return "unsufficient"


def is_already_assessed(doc: Dict[str, Any]) -> bool:
    rag = doc.get("rag_quality")
    return isinstance(rag, dict) and rag.get("score") is not None


def assess_document_rag(doc: Dict[str, Any], max_retries: int = 5) -> Dict[str, Any]:
    prompt = rag_filter_prompt(doc)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": """
                            You are an expert evaluator for Azerbaijani RAG benchmark construction.

                            Your job is NOT to be generous.
                            Your job is to be discriminative.

                            Important scoring behavior:
                            - Use low values often when the document is weak.
                            - Avoid clustering everything in the upper range.
                            - Many real-world documents should be mediocre or poor for RAG.
                            - Only assign high values if the document is clearly rich, self-contained, and QA-friendly.

                            Return strict JSON only.
                        """
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )

            result_text = response.choices[0].message.content
            parsed = parse_llm_response(result_text)

            signals = parsed.get("signals", {})
            score = compute_rag_score(signals)
            keep_priority = derive_keep_priority(score)

            parsed["score"] = score
            parsed["keep_priority"] = keep_priority

            return parsed

        except Exception as e:
            wait = 2 ** attempt
            print(f"API error (attempt {attempt+1}/{max_retries}) for doc_id={doc.get('doc_id', 'N/A')}: {e}")
            time.sleep(wait)

    return None


def load_json_file(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(path: str, data: List[Dict[str, Any]]):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def load_resume_dataset(input_file: str, output_file: str) -> List[Dict[str, Any]]:
    if os.path.exists(output_file):
        print(f"Resume mode: existing output found -> {output_file}")
        return load_json_file(output_file)
    else:
        print(f"Fresh start: loading input -> {input_file}")
        return load_json_file(input_file)


# PARALLEL PROCESSING
def process_documents_parallel(
    data: List[Dict[str, Any]],
    max_workers: int = 6,
    checkpoint_interval: int = 50,
    output_file: str = "../data/dataset_evaluated.json"
):
    pending_indices = [i for i, doc in enumerate(data) if not is_already_assessed(doc)]

    if not pending_indices:
        print("Everything is already assessed. Nothing to do.")
        return

    def worker(idx: int):
        doc = data[idx]
        doc["rag_quality"] = assess_document_rag(doc)
        return idx, doc

    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, idx): idx for idx in pending_indices}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Assessing remaining docs"):
            idx = futures[future]

            try:
                result_idx, updated_doc = future.result()
                data[result_idx] = updated_doc
            except Exception as e:
                print(f"Worker failed at index {idx}: {e}")

            completed += 1

            if completed % checkpoint_interval == 0:
                with write_lock:
                    save_json_file(output_file, data)

    with write_lock:
        save_json_file(output_file, data)

    print(f"\nProcess completed. Results saved to {output_file}.")

# If you do not find "../data/dataset.json" and "../data/dataset_evaluated.json", probably, they are uploaded to HuggingFace
def main(input_file: str = "../data/dataset.json", output_file: str = "../data/dataset_evaluated.json"):
    if not os.path.exists(input_file) and not os.path.exists(output_file):
        print(f"Error: neither {input_file} nor {output_file} exists!")
        return

    data = load_resume_dataset(input_file, output_file)

    print(f"{len(data)} documents loaded. Starting/resuming assessment...")

    process_documents_parallel(
        data=data,
        max_workers=6,
        checkpoint_interval=50,
        output_file=output_file
    )


if __name__ == "__main__":
    main()