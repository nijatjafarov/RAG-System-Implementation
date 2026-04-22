import json
import re
from typing import List, Dict

MAX_CHARS = 1000
MIN_CHARS = 300
OVERLAP_CHARS = 200

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def split_long_text(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP_CHARS) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks = []
    current = ""

    for p in paragraphs:
        if len(current) + len(p) + 2 <= max_chars:
            current += ("\n\n" if current else "") + p
        else:
            if current:
                chunks.append(current.strip())
            current = p

    if current:
        chunks.append(current.strip())

    # If some chunks are still too long
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
        else:
            start = 0
            while start < len(chunk):
                end = min(start + max_chars, len(chunk))
                final_chunks.append(chunk[start:end].strip())
                if end == len(chunk):
                    break
                start = max(end - overlap, 0)

    return [c for c in final_chunks if len(c.strip()) >= MIN_CHARS]


def merge_small_chunks(chunks: List[str], min_chars: int = MIN_CHARS, max_chars: int = MAX_CHARS) -> List[str]:
    merged = []
    current = ""

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        if len(chunk) >= min_chars:
            if current:
                if len(current) + len(chunk) + 2 <= max_chars:
                    current += "\n\n" + chunk
                    merged.append(current.strip())
                    current = ""
                else:
                    merged.append(current.strip())
                    current = ""
                    merged.append(chunk)
            else:
                merged.append(chunk)
        else:
            if not current:
                current = chunk
            elif len(current) + len(chunk) + 2 <= max_chars:
                current += "\n\n" + chunk
            else:
                merged.append(current.strip())
                current = chunk

    if current:
        if merged and len(current) < min_chars and len(merged[-1]) + len(current) + 2 <= max_chars:
            merged[-1] += "\n\n" + current
        else:
            merged.append(current.strip())

    return [m for m in merged if m.strip()]


# Domain-specific splitters
def split_legal(text: str) -> List[str]:
    """
    Cheap structure-aware legal chunking for Azerbaijani legal docs.
    Priority:
    1. Maddə / Fəsil / Bölmə / numbered clauses
    2. fallback paragraph grouping
    """
    text = normalize_text(text)

    legal_markers = [
        r'(?=\bMaddə\s+\d+[\.\-]?)',
        r'(?=\bMADDƏ\s+\d+[\.\-]?)',
        r'(?=\bFəsil\s+[IVXLC\d]+[\.\-]?)',
        r'(?=\bFƏSİL\s+[IVXLC\d]+[\.\-]?)',
        r'(?=\bBölmə\s+[IVXLC\d]+[\.\-]?)',
        r'(?=\bBÖLMƏ\s+[IVXLC\d]+[\.\-]?)',
        r'(?=\n\d+\.\s)',          # 1.
        r'(?=\n\d+\.\d+\.\s?)',    # 1.1.
    ]

    for pattern in legal_markers:
        parts = re.split(pattern, text)
        parts = [p.strip() for p in parts if p.strip()]
        meaningful = [p for p in parts if len(p) >= MIN_CHARS]

        if len(meaningful) >= 2:
            output = []
            for part in meaningful:
                if len(part) > MAX_CHARS:
                    output.extend(split_long_text(part))
                else:
                    output.append(part)
            return merge_small_chunks(output)

    return merge_small_chunks(split_long_text(text))


def split_news(text: str) -> List[str]:
    text = normalize_text(text)
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    if not paragraphs:
        return split_long_text(text)

    chunks = []
    current = ""

    for p in paragraphs:
        if len(current) + len(p) + 2 <= MAX_CHARS:
            current += ("\n\n" if current else "") + p
        else:
            if current:
                chunks.append(current.strip())
            current = p

    if current:
        chunks.append(current.strip())

    return merge_small_chunks(chunks)


def split_encyclopedic(text: str) -> List[str]:
    text = normalize_text(text)

    sections = re.split(r'\n(?=[A-ZƏÖÜÇŞİI][^\n]{0,80}\n)', text)
    sections = [s.strip() for s in sections if s.strip()]

    if len(sections) >= 2:
        output = []
        for sec in sections:
            if len(sec) > MAX_CHARS:
                output.extend(split_long_text(sec))
            else:
                output.append(sec)
        return merge_small_chunks(output)

    return merge_small_chunks(split_long_text(text))


def chunk_document(doc: Dict) -> List[Dict]:
    doc_id = doc.get("doc_id", "")
    title = doc.get("title", "")
    domain = doc.get("domain", "").lower()
    text = doc.get("text", "")

    if not text.strip():
        return []

    if domain == "legal":
        chunks = split_legal(text)
    elif domain == "news":
        chunks = split_news(text)
    else:
        chunks = split_encyclopedic(text)

    output = []
    for i, chunk_text in enumerate(chunks, 1):
        embedding_text = f"""Title: {title}
Domain: {domain}
Source: {doc.get("source", "")}
Published at: {doc.get("published_at", "")}

Text:
{chunk_text}""".strip()
        
        output.append({
            "chunk_id": f"{doc_id}_chunk_{i}",
            "doc_id": doc_id,
            "title": title,
            "domain": domain,
            "source": doc.get("source", ""),
            "url": doc.get("url", ""),
            "published_at": doc.get("published_at", ""),
            "corpus_tier": doc.get("corpus_tier", ""),
            "text": chunk_text,
            "embedding_text": embedding_text,
            "length": len(embedding_text)
        })

    return output


def chunk_corpus(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    all_chunks = []

    for doc in docs:
        doc_chunks = chunk_document(doc)
        all_chunks.extend(doc_chunks)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Done. {len(all_chunks)} chunks saved to {output_path}")


if __name__ == "__main__":
    chunk_corpus("../data/main_corpus.json", "../data/main_chunks.json")