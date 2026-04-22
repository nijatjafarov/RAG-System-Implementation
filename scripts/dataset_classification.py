import json

MAIN_THRESHOLD = 80
ARCHIVE_THRESHOLD = 50

def classify_document(doc):
    rq = doc.get("rag_quality", {})
    score = rq.get("score", 0)

    if score >= MAIN_THRESHOLD:
        return "main"

    if score >= ARCHIVE_THRESHOLD:
        return "archive"

    return "drop"


def split_corpora(docs):
    main_corpus = []
    archive_corpus = []
    dropped_corpus = []

    for doc in docs:
        category = classify_document(doc)

        doc["corpus_tier"] = category

        if category == "main":
            main_corpus.append(doc)
        elif category == "archive":
            archive_corpus.append(doc)
        else:
            dropped_corpus.append(doc)

    return main_corpus, archive_corpus, dropped_corpus

if __name__ == "__main__":
    with open("../data/dataset_evaluated.json", "r", encoding="utf-8") as f:
        assessed_docs = json.load(f)

    main_corpus, archive_corpus, dropped_corpus = split_corpora(assessed_docs)

    with open("../data/main_corpus.json", "w", encoding="utf-8") as f:
        json.dump(main_corpus, f, ensure_ascii=False, indent=2)

    with open("../data/archive_corpus.json", "w", encoding="utf-8") as f:
        json.dump(archive_corpus, f, ensure_ascii=False, indent=2)

    with open("../data/dropped_corpus.json", "w", encoding="utf-8") as f:
        json.dump(dropped_corpus, f, ensure_ascii=False, indent=2)