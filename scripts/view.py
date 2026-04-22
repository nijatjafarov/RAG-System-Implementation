import streamlit as st
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict

st.set_page_config(page_title="RAG Question Analysis", layout="wide")
st.title("RAG Question Analysis")

@st.cache_data
def load_questions(file_path: str) -> List[Dict]:
    path = Path(file_path)
    if not path.exists():
        st.error(f"File not found: {file_path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        raw_questions = data
    elif isinstance(data, dict):
        if "question_id" in data and "question" in data:
            raw_questions = [data]
        else:
            raw_questions = list(data.values())
    else:
        return []

    merged = {}
    for q in raw_questions:
        qid = q["question_id"]
        if qid not in merged:
            merged[qid] = q.copy()
        else:
            existing_combs = merged[qid].get("combinations", {})
            new_combs = q.get("combinations", {})
            merged[qid]["combinations"] = {**existing_combs, **new_combs}
    return list(merged.values())

questions = load_questions("data/eval_results_detailed.json")
if not questions:
    st.stop()

@st.cache_data
def load_chunks(file_path: str) -> Dict[str, str]:
    path = Path(file_path)
    if not path.exists():
        st.error(f"Chunks file not found: {file_path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        chunks_dict = {}
        for item in data:
            if isinstance(item, dict) and 'chunk_id' in item and 'text' in item:
                chunks_dict[item['chunk_id']] = item['text']
            else:
                st.warning(f"Skipping chunk item missing 'chunk_id' or 'text': {item}")
        return chunks_dict
    elif isinstance(data, dict):
        return data
    else:
        st.error(f"Unexpected chunks file format: {type(data)}")
        return {}

question_options = {q["question_id"]: q["question"] for q in questions}
selected_qid = st.selectbox(
    "Select a Question",
    options=list(question_options.keys()),
    format_func=lambda x: f"{question_options[x]}"
)

current_q = next(q for q in questions if q["question_id"] == selected_qid)


# Question and gold answer
st.divider()
st.header("Question & Gold Answer")

st.markdown(f"**Question:** {current_q['question']}")
st.markdown(f"**Gold Answer:** {current_q['gold_answer']}")

golden_relevant_ids = current_q.get("context_chunk_ids", [])
golden_relevant_chunks = []

chunks = load_chunks('data/main_chunks.json')

for golden_relevant_id in golden_relevant_ids:
    golden_relevant_chunks.append(chunks.get(golden_relevant_id))
    
# Display relevant chunks
st.subheader("Relevant Context")
if golden_relevant_chunks:
    for text in golden_relevant_chunks:
        st.markdown(text)
else:
    st.info("No relevant chunks found in this question.")


# Combinations
combinations = current_q.get("combinations", {})
if not combinations:
    st.warning("No combinations found for this question.")
    st.stop()

# Evaluation metrics table
st.divider()
st.header("Evaluation Metrics per Combination")

rows = []
for combo_name, combo in combinations.items():
    eval_data = combo.get("evaluation", {})
    retrieval_data = combo.get("retrieval", {})
    gen = combo.get("generation", {})
    
    rows.append({
        "Combination": combo_name,
        "MRR": retrieval_data.get("mrr"),
        "Recall@5": retrieval_data.get("recall@5"),
        "NDCG@5": retrieval_data.get("ndcg@5"),
        "Generated Answer": gen.get("generated_answer", ""),
        "Semantic Similarity": eval_data.get("semantic_similarity"),
        "LLM Judge Score": eval_data.get("llm_judge_score")
    })

df = pd.DataFrame(rows)
st.dataframe(
    df,
    column_config={
        "MRR": st.column_config.NumberColumn(format="%.4f"),
        "Recall@5": st.column_config.NumberColumn(format="%.4f"),
        "NDCG@5": st.column_config.NumberColumn(format="%.4f"),
        "Generated Answer": st.column_config.TextColumn(width="large"),
        "Semantic Similarity": st.column_config.NumberColumn(format="%.4f"),
        "LLM Judge Score": st.column_config.NumberColumn(format="%.2f")
    },
    use_container_width=True,
    hide_index=True
)