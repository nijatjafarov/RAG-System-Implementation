import streamlit as st
from pinecone import Pinecone
import google.genai as genai
import numpy as np
import time
from openai import OpenAI
import os

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "azrag-gemini-embed2"
NAMESPACE = "main"
TOP_K = 5


@st.cache_resource
def init_clients():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    genai_client = genai.Client(api_key=GEMINI_API_KEY)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return index, genai_client, openai_client

index, genai_client, openai_client = init_clients()

RAG_SYSTEM = """Answer based on the provided context."
Answer in Azerbaijani. Be concise."""

ABSTENTION_PHRASE = "verilmiş məlumatlar əsasında bu suala cavab vermək mümkün deyil"


def is_abstention(text: str) -> bool:
    return ABSTENTION_PHRASE in (text or "").lower()

def embed_query(query: str):
    result = genai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=[query],
        config=genai.types.EmbedContentConfig(task_type="retrieval_query"),
    )
    return result.embeddings[0].values

def retrieve(query: str, top_k: int = TOP_K):
    t0 = time.time()
    vec = embed_query(query)
    res = index.query(vector=vec, top_k=top_k, include_metadata=True, namespace=NAMESPACE)
    latency = time.time() - t0
    passages = []
    for m in res.matches:
        text = m.metadata.get("text") or m.metadata.get("embedding_text", "")
        passages.append({"id": m.id, "score": m.score, "text": text})
    return passages, latency

def build_rag_prompt(question: str, passages: list) -> str:
    ctx = "\n\n".join(
        f"[Passage {i+1}]\n{p['text'][:1500]}" for i, p in enumerate(passages)
    )
    return f"Context:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"

def generate_rag(question: str, passages: list):
    t0 = time.time()
    
    response = openai_client.chat.completions.create(
        model="gpt-5.4",
        messages=[
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": build_rag_prompt(question, passages)}
        ],
        temperature=0.7,
        max_completion_tokens=500
    )
    return response.choices[0].message.content, time.time() - t0

def generate_llm_only(question: str):
    t0 = time.time()
    response = openai_client.chat.completions.create(
        model="gpt-5.4",
        messages=[
            {"role": "user", "content": question}
        ],
        temperature=0.7,
        max_completion_tokens=500
    )
    return response.choices[0].message.content, time.time() - t0


def cosine_sim_gemini(text_a: str, text_b: str) -> float:
    if not text_a or not text_b:
        return 0.0
    result = genai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=[text_a, text_b],
        config=genai.types.EmbedContentConfig(task_type="semantic_similarity"),
    )
    v1 = np.array(result.embeddings[0].values)
    v2 = np.array(result.embeddings[1].values)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.clip(np.dot(v1, v2) / denom, 0.0, 1.0)) if denom > 0 else 0.0

def avg_chunk_score(passages: list) -> float:
    scores = [p["score"] for p in passages]
    return float(np.mean(scores)) if scores else 0.0

def max_chunk_score(passages: list) -> float:
    return max((p["score"] for p in passages), default=0.0)

def score_bar(value: float, color: str = "#1D9E75") -> str:
    pct = int(value * 100)
    return (
        f'<div style="background:#e5e7eb;border-radius:4px;height:8px;width:100%">'
        f'<div style="background:{color};width:{pct}%;height:8px;border-radius:4px"></div>'
        f'</div>'
        f'<small style="color:#6b7280">{value:.3f}</small>'
    )


st.set_page_config(page_title="AzRAG — RAG vs LLM", layout="wide", page_icon="🇦🇿")

st.markdown("""
<style>
.metric-card {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}
.metric-val { font-size: 1.6rem; font-weight: 600; margin: 0; }
.metric-lbl { font-size: 0.72rem; color: #6b7280; margin-top: 2px; text-transform: uppercase; letter-spacing: .05em; }
.answer-box {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 14px 16px;
    min-height: 120px;
    font-size: 0.95rem;
    line-height: 1.7;
}
.abstain-box {
    background: #fffbeb;
    border: 1px solid #fbbf24;
    border-radius: 10px;
    padding: 10px 14px;
    color: #92400e;
    font-size: 0.88rem;
}
.section-head {
    font-size: 0.78rem;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: .06em;
    margin-bottom: 6px;
}
.tag-rag  { background:#d1fae5; color:#065f46; border-radius:6px; padding:2px 8px; font-size:.75rem; font-weight:600; }
.tag-llm  { background:#ede9fe; color:#3730a3; border-radius:6px; padding:2px 8px; font-size:.75rem; font-weight:600; }
.tag-abs  { background:#fef3c7; color:#92400e; border-radius:6px; padding:2px 8px; font-size:.75rem; font-weight:600; }
div[data-testid="stExpander"] { border: 1px solid #e5e7eb !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


st.markdown("## 🇦🇿 AzRAG — RAG vs LLM Comparison")
st.caption("Gemini Embedding · Pinecone · GPT-5.4 &nbsp;|&nbsp; Metrics: semantic similarity, retrieval scores, latency")
st.divider()


with st.form("query_form"):
    col_q, col_k = st.columns([5, 1])
    with col_q:
        query = st.text_input("Sualınızı daxil edin / Enter your question:", placeholder="Sual yazın...")
    with col_k:
        top_k = st.number_input("Top-K", min_value=1, max_value=20, value=TOP_K, step=1)

    ref_answer = st.text_input(
        "İstinad cavabı (optional — for semantic similarity scoring):",
        placeholder="Leave blank to compare RAG vs LLM answers directly",
    )
    submitted = st.form_submit_button("▶  Cavab ver", use_container_width=True, type="primary")


if submitted:
    if not query.strip():
        st.warning("Zəhmət olmasa sual daxil edin.")
        st.stop()

    with st.spinner("Retrieving chunks & generating answers…"):
        passages, ret_latency = retrieve(query, top_k)
        rag_answer, rag_latency = generate_rag(query, passages)
        llm_answer, llm_latency = generate_llm_only(query)

    rag_abstained = is_abstention(rag_answer)
    llm_abstained = is_abstention(llm_answer)


    with st.spinner("Computing semantic similarity…"):
        if ref_answer.strip():
            sim_rag = cosine_sim_gemini(rag_answer, ref_answer)
            sim_llm = cosine_sim_gemini(llm_answer, ref_answer)
            sim_label = "vs reference answer"
        else:
            sim_cross = cosine_sim_gemini(rag_answer, llm_answer)
            sim_rag = sim_cross
            sim_llm = sim_cross
            sim_label = "RAG ↔ LLM (cross-similarity)"

    scores = [p["score"] for p in passages]
    avg_score = avg_chunk_score(passages)
    max_score = max_chunk_score(passages)

    st.divider()


    col_rag, col_llm = st.columns(2, gap="medium")

    with col_rag:
        st.markdown('<p class="section-head">RAG answer <span class="tag-rag">RAG + Pinecone</span></p>', unsafe_allow_html=True)
        if rag_abstained:
            st.markdown(f'<div class="abstain-box">⚠️ Abstained — context insufficient</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-box">{rag_answer}</div>', unsafe_allow_html=True)

    with col_llm:
        st.markdown('<p class="section-head">LLM-only answer <span class="tag-llm">GPT-5.4 Baseline</span></p>', unsafe_allow_html=True)
        if llm_abstained:
            st.markdown(f'<div class="abstain-box">⚠️ Abstained</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-box">{llm_answer}</div>', unsafe_allow_html=True)

    st.divider()


    st.markdown("### 📊 Metrics")

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
          <p class="metric-val" style="color:#1D9E75">{sim_rag:.3f}</p>
          <p class="metric-lbl">RAG sem-sim<br><span style="color:#aaa;font-size:.65rem">{sim_label}</span></p>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
          <p class="metric-val" style="color:#7F77DD">{sim_llm:.3f}</p>
          <p class="metric-lbl">LLM sem-sim<br><span style="color:#aaa;font-size:.65rem">{sim_label}</span></p>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
          <p class="metric-val">{max_score:.3f}</p>
          <p class="metric-lbl">Max chunk<br>score</p>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
          <p class="metric-val">{avg_score:.3f}</p>
          <p class="metric-lbl">Avg chunk<br>score</p>
        </div>""", unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="metric-card">
          <p class="metric-val">{rag_latency + ret_latency:.2f}s</p>
          <p class="metric-lbl">RAG total<br>latency</p>
        </div>""", unsafe_allow_html=True)

    with c6:
        st.markdown(f"""
        <div class="metric-card">
          <p class="metric-val">{llm_latency:.2f}s</p>
          <p class="metric-lbl">GPT-5.4<br>latency</p>
        </div>""", unsafe_allow_html=True)


    if rag_abstained or llm_abstained:
        st.markdown("")
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            label = "🚫 RAG abstained" if rag_abstained else "✅ RAG answered"
            color = "#fef3c7" if rag_abstained else "#d1fae5"
            st.markdown(f'<div style="background:{color};border-radius:8px;padding:8px 12px;font-size:.85rem">{label}</div>', unsafe_allow_html=True)
        with fcol2:
            label = "🚫 LLM abstained" if llm_abstained else "✅ LLM answered"
            color = "#fef3c7" if llm_abstained else "#d1fae5"
            st.markdown(f'<div style="background:{color};border-radius:8px;padding:8px 12px;font-size:.85rem">{label}</div>', unsafe_allow_html=True)

    st.divider()
    

    st.markdown("### 🔍 Retrieval insight")

    ri_col1, ri_col2, ri_col3 = st.columns(3)
    with ri_col1:
        st.metric("Chunks retrieved", len(passages))
    with ri_col2:
        st.metric("Retrieval latency", f"{ret_latency:.2f}s")
    with ri_col3:
        st.metric("Score range", f"{min(scores):.3f} – {max(scores):.3f}" if scores else "—")


    if scores:
        import pandas as pd
        df = pd.DataFrame({"Passage": [f"P{i+1}" for i in range(len(scores))], "Score": scores})
        st.bar_chart(df.set_index("Passage"), height=160, color="#1D9E75")


    st.markdown('<p class="section-head" style="margin-top:8px">Retrieved passages</p>', unsafe_allow_html=True)
    for i, p in enumerate(passages):
        with st.expander(f"Passage {i+1} · id: {p['id']} · score: {p['score']:.4f}"):
            st.progress(float(np.clip(p["score"], 0, 1)), text=f"{p['score']:.4f}")
            st.write(p["text"])