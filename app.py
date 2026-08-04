import os
import time
import sys
from io import BytesIO
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, List, Dict, Any
from collections import defaultdict

# Adjust path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import rag_backend

# Set page config
st.set_page_config(
    page_title="AI Research Paper Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Glassmorphism and animations
st.markdown(
    """
    <style>
    /* Dark Theme Core overrides */
    .stApp {
        background-color: #0b0d10;
        color: #e2e8f0;
    }
    
    /* Glassmorphic Containers */
    .glass-card {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 12px 40px 0 rgba(99, 102, 241, 0.1);
    }
    
    /* Header Gradient */
    .gradient-header {
        background: linear-gradient(135deg, #a78bfa 0%, #6366f1 50%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem !important;
        margin-bottom: 5px;
    }
    
    .gradient-subtitle {
        color: #94a3b8;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d0f13 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Custom buttons with gradients */
    .stButton button {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(59, 130, 246, 0.2) 100%) !important;
        border: 1px solid rgba(99, 102, 241, 0.4) !important;
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #6366f1 0%, #3b82f6 100%) !important;
        color: white !important;
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.6) !important;
        border-color: transparent !important;
        transform: translateY(-1px);
    }
    
    /* Custom input forms */
    .stTextInput input {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
    }
    
    .stTextInput input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 10px rgba(99, 102, 241, 0.3) !important;
    }
    
    /* Source item cards */
    .source-card {
        background: rgba(99, 102, 241, 0.02);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
        transition: all 0.2s ease;
    }
    
    .source-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        background: rgba(99, 102, 241, 0.04);
    }
    
    /* Badges */
    .badge-similarity {
        background-color: rgba(99, 102, 241, 0.2);
        color: #a78bfa;
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-meta {
        background-color: rgba(255, 255, 255, 0.05);
        color: #cbd5e1;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        display: inline-block;
        margin-right: 5px;
    }
    
    /* Pipeline graph visualization CSS */
    .pipeline-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin: 20px 0;
    }
    
    .pipeline-node {
        width: 250px;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        margin: 5px 0;
    }
    
    .pipeline-node.active {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(59, 130, 246, 0.2) 100%);
        border: 2px solid #6366f1;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.5);
        transform: scale(1.05);
        animation: pulse 1.5s infinite alternate;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 10px rgba(99, 102, 241, 0.3); }
        100% { box-shadow: 0 0 25px rgba(99, 102, 241, 0.8); }
    }
    
    .pipeline-arrow {
        font-size: 20px;
        color: #4b5563;
        margin: 2px 0;
        font-weight: bold;
    }
    
    .pipeline-arrow.active {
        color: #6366f1;
        animation: arrow-glow 1s infinite alternate;
    }
    
    @keyframes arrow-glow {
        0% { transform: translateY(0); color: #4b5563; }
        100% { transform: translateY(3px); color: #6366f1; }
    }
    
    /* Styled Chat bubbles */
    .user-bubble {
        background-color: rgba(99, 102, 241, 0.1) !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 12px 12px 0 12px !important;
        padding: 15px;
        margin-bottom: 15px;
    }
    
    .assistant-bubble {
        background-color: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px 12px 12px 0 !important;
        padding: 15px;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize Session State Variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List of tuples (User, Assistant)
if "last_query" not in st.session_state:
    st.session_state.last_query = None
if "last_results" not in st.session_state:
    st.session_state.last_results = None  # Results of active retrieval method
if "all_methods_results" not in st.session_state:
    st.session_state.all_methods_results = {
        "Vector Search": None,
        "Hybrid": None
    }
if "pipeline_stage" not in st.session_state:
    st.session_state.pipeline_stage = "Idle"
if "query_stats_history" not in st.session_state:
    st.session_state.query_stats_history = []  # Logs for analytics
if "llm_model" not in st.session_state:
    st.session_state.llm_model = "gpt-4o"
if "embedding_model_name" not in st.session_state:
    st.session_state.embedding_model_name = "text-embedding-3-small"

# Error Handling Wrappers & Status Checkers
def verify_environment() -> Tuple[bool, List[str]]:
    """Verifies that all API keys and directories are ready."""
    warnings = []
    ready = True
    
    # Check .env configuration
    if not os.environ.get("OPENAI_API_KEY"):
        warnings.append("⚠️ **OpenAI API Key is missing** in `.env`. Chat operations will fail.")
        ready = False
        
    cohere_key = os.environ.get("COHERE_API_KEY") or os.environ.get("CO_API_KEY")
    if not cohere_key:
        warnings.append("ℹ️ **Cohere API Key is missing** in `.env`. The 'Reranker' retrieval method will be disabled.")
        
    if not os.path.exists("docs"):
        os.makedirs("docs", exist_ok=True)
        warnings.append("📁 Created missing `docs/` folder. Place your reference documents inside.")
        
    db_exists = os.path.exists("db/chroma_db")
    if not db_exists:
        warnings.append("⚠️ **Vector Database not found**. Please upload a PDF and click **Build Database** in the sidebar to initialize it.")
        
    return ready, warnings

# Main RAG execution loop
def execute_query(query_text: str, retrieval_method: str, k: int, temperature: float):
    """Executes the full conversational RAG flow and caches state."""
    # 1. Update pipeline state
    st.session_state.pipeline_stage = "History Query Rewrite"
    
    # Rewrite standalone query if memory exists
    standalone_query = rag_backend.rewrite_query_with_history(query_text, st.session_state.chat_history, st.session_state.llm_model)
    
    # 2. Document Loading / Embeddings / Retrieval
    st.session_state.pipeline_stage = "Retrieving Documents"
    retrieval_output = rag_backend.perform_retrieval(standalone_query, retrieval_method, k)
    
    st.session_state.last_query = query_text
    st.session_state.last_results = retrieval_output
    
    # Pre-populate active retrieval method in all_methods_results
    st.session_state.all_methods_results[retrieval_method] = retrieval_output
    
    # Invalidate other methods so they lazy load when comparison page is opened
    for method in st.session_state.all_methods_results:
        if method != retrieval_method:
            st.session_state.all_methods_results[method] = None
            
    # 3. LLM Response Generation
    st.session_state.pipeline_stage = "LLM Answering"
    
    # Generate streamed answer
    return standalone_query, retrieval_output["documents"]

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.image("https://img.icons8.com/nolan/96/artificial-intelligence.png", width=64)
    st.markdown('<h2 style="margin-top:10px;">RAG Settings</h2>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Upload PDF Section
    st.markdown("### 📄 Document Upload")
    uploaded_pdf = st.file_uploader("Upload PDF Paper", type=["pdf"])
    if uploaded_pdf is not None:
        if st.button("Ingest & Convert to TXT"):
            with st.spinner("Extracting text and saving to docs/ ..."):
                try:
                    txt_path = rag_backend.save_uploaded_pdf(uploaded_pdf)
                    st.success(f"Extracted to `{txt_path}`!")
                except Exception as e:
                    st.error(f"Failed to parse PDF: {str(e)}")
                    
    st.markdown("---")
    
    # Database Actions
    st.markdown("### 🗄️ Database Operations")
    col_db1, col_db2 = st.columns(2)
    with col_db1:
        if st.button("Build Database"):
            with st.spinner("Indexing documents..."):
                try:
                    msg = rag_backend.run_db_ingestion()
                    st.success(msg)
                except Exception as e:
                    st.error(f"Failed to build DB: {str(e)}")
    with col_db2:
        if st.button("Reload Database"):
            rag_backend.reset_db_cache()
            st.success("Cache cleared!")
            
    st.markdown("---")
    
    # Settings Sliders & Selectors
    st.markdown("### ⚙️ Parameters")
    
    # Available retrieval methods selection
    method_options = ["Vector Search", "Hybrid"]
    
    retrieval_method = st.radio(
        "Choose Retrieval Method",
        method_options,
        index=0
    )
    
    top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=20, value=5)
    temperature = st.slider("LLM Temperature", min_value=0.0, max_value=2.0, value=0.2, step=0.1)
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.last_query = None
        st.session_state.last_results = None
        st.success("Cleared!")

# ----------------- MAIN PAGE -----------------
# Header Section
st.markdown('<h1 class="gradient-header">AI Research Paper Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="gradient-subtitle">Retrieval-Augmented Generation using LangChain + ChromaDB + Hybrid Retrieval</p>', unsafe_allow_html=True)

# Environment Readiness Checklist
env_ok, env_warnings = verify_environment()
for warn in env_warnings:
    st.warning(warn)

# Create Pages inside Horizontal Navigation Tabs
tab_chat, tab_compare, tab_analytics, tab_pipeline, tab_logs = st.tabs([
    "💬 AI Assistant", 
    "🔍 Retrieval Comparison", 
    "📊 Analytics", 
    "Tracks Pipeline", 
    "📝 Execution Logs"
])

# ----------------- PAGE 1: AI ASSISTANT -----------------
with tab_chat:
    # Historical Chat display
    for user_q, ai_ans in st.session_state.chat_history:
        st.markdown(f'<div class="user-bubble">🧑‍💻 <b>User:</b><br>{user_q}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="assistant-bubble">🤖 <b>AI Assistant:</b><br>{ai_ans}</div>', unsafe_allow_html=True)
        
    # Input Area
    user_query = st.text_input("Ask a question about your research papers:", key="user_query_input")
    
    if st.button("🚀 Search & Answer"):
        if not env_ok:
            st.error("Cannot run query. Check warnings and configure API keys.")
        elif not user_query:
            st.warning("Please enter a question.")
        else:
            # Layout container for status
            status_container = st.empty()
            status_container.info("🔄 Re-writing query with context history...")
            
            # Execute pipeline
            try:
                standalone_q, docs = execute_query(user_query, retrieval_method, top_k, temperature)
                
                # Render standalone query if re-written
                if standalone_q != user_query:
                    st.caption(f"🔎 Rewritten search terms: *\"{standalone_q}\"*")
                
                status_container.info("📝 Streaming response from LLM...")
                
                # Render User question bubble
                st.markdown(f'<div class="user-bubble">🧑‍💻 <b>User:</b><br>{user_query}</div>', unsafe_allow_html=True)
                
                # Setup generator streaming response container
                response_box = st.empty()
                full_ans = ""
                
                ans_generator = rag_backend.generate_rag_answer_stream(
                    query=standalone_q,
                    retrieved_docs=docs,
                    chat_history=st.session_state.chat_history,
                    temperature=temperature,
                    llm_model=st.session_state.llm_model
                )
                
                for chunk in ans_generator:
                    full_ans += chunk
                    response_box.markdown(f'<div class="assistant-bubble">🤖 <b>AI Assistant:</b><br>{full_ans}</div>', unsafe_allow_html=True)
                
                # Finish answering
                st.session_state.pipeline_stage = "Idle"
                status_container.empty()
                
                # Add to chat history
                st.session_state.chat_history.append((user_query, full_ans))
                
                # Log metrics for analytics
                st.session_state.query_stats_history.append({
                    "timestamp": time.time(),
                    "query": user_query,
                    "method": retrieval_method,
                    "k": top_k,
                    "retrieval_time": st.session_state.last_results["retrieval_time"],
                    "avg_similarity": sum(st.session_state.last_results["similarity_scores"]) / len(docs) if docs else 0,
                    "num_docs": len(docs)
                })
                
                # Forces rerun to render buttons and static states cleanly
                st.rerun()
                
            except Exception as e:
                status_container.empty()
                st.session_state.pipeline_stage = "Idle"
                st.error(f"Execution Error: {str(e)}")

    # Post-answering details (Sources & citations)
    if st.session_state.last_results:
        st.markdown("---")
        st.subheader("📌 Retrieval Summary")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Retrieved Documents", len(st.session_state.last_results["documents"]))
        col_m2.metric("Retrieval Latency", f"{st.session_state.last_results['retrieval_time']:.3f} s")
        avg_score = sum(st.session_state.last_results["similarity_scores"]) / len(st.session_state.last_results["similarity_scores"]) if st.session_state.last_results["similarity_scores"] else 0.0
        col_m2_status = col_m3.metric("Average Similarity", f"{avg_score:.2%}")
        
        # Download and copy buttons
        if st.session_state.chat_history:
            last_ans = st.session_state.chat_history[-1][1]
            st.download_button(
                label="📥 Download Answer",
                data=last_ans,
                file_name="answer.txt",
                mime="text/plain"
            )
            
        # Display expandable source chunks
        st.markdown("### 📚 Source Chunks used in Generation")
        
        for idx, doc in enumerate(st.session_state.last_results["documents"]):
            meta = rag_backend.parse_chunk_metadata(doc, idx)
            sim_score = st.session_state.last_results["similarity_scores"][idx]
            raw_score = st.session_state.last_results["raw_scores"][idx]
            
            with st.expander(f"📄 [{idx+1}] {meta['doc_name']} (Page {meta['page_number']}) - Score: {sim_score:.2%}"):
                st.markdown(f'<div class="badge-similarity">Similarity: {sim_score:.2%}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="badge-meta">Raw Score: {raw_score:.4f}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="badge-meta">ID: {meta["chunk_id"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="badge-meta">Source: {meta["source_path"]}</div>', unsafe_allow_html=True)
                st.markdown("---")
                st.markdown(f"*{doc.page_content}*")

# ----------------- PAGE 2: RETRIEVAL COMPARISON -----------------
with tab_compare:
    st.subheader("🔍 Retrieval Method Comparison Matrix")
    
    if not st.session_state.last_query:
        st.info("Ask a question first to compare different retrieval methods side-by-side.")
    else:
        st.markdown(f"**Comparing results for last query:** *\"{st.session_state.last_query}\"*")
        
        # Implement Lazy Loading for other retrieval methods
        with st.spinner("Lazy loading retrieval methods..."):
            for m in st.session_state.all_methods_results:
                if st.session_state.all_methods_results[m] is None:
                    try:
                        # Lazy load documents for comparison
                        st.session_state.all_methods_results[m] = rag_backend.perform_retrieval(
                            query=st.session_state.last_query,
                            method=m,
                            k=top_k
                        )
                    except Exception as e:
                        st.session_state.all_methods_results[m] = {"error": str(e)}
                        
        # Render a neat table comparison
        rows = []
        for method, res in st.session_state.all_methods_results.items():
            if not res or "error" in res:
                error_msg = res.get("error", "Not computed") if res else "Not computed"
                rows.append({
                    "Method": method,
                    "Retrieved": 0,
                    "Avg Similarity": "N/A",
                    "Latency": "N/A",
                    "Status": f"❌ Error: {error_msg}"
                })
                continue
                
            scores = res["similarity_scores"]
            avg_sim = sum(scores) / len(scores) if scores else 0.0
            rows.append({
                "Method": method,
                "Retrieved": len(res["documents"]),
                "Avg Similarity": f"{avg_sim:.2%}",
                "Latency": f"{res['retrieval_time'] * 1000:.1f} ms",
                "Status": "✅ Success"
            })
            
        st.table(pd.DataFrame(rows))
        
        # Display side-by-side details
        compare_cols = st.columns(len(st.session_state.all_methods_results))
        for i, (method, res) in enumerate(st.session_state.all_methods_results.items()):
            with compare_cols[i]:
                st.markdown(f"#### 🌐 {method}")
                if not res or "error" in res:
                    st.caption("No results/error")
                    continue
                    
                st.caption(f"Latency: {res['retrieval_time']*1000:.1f} ms")
                for doc_idx, doc in enumerate(res["documents"][:5]): # Show top 5 max
                    meta = rag_backend.parse_chunk_metadata(doc, doc_idx)
                    score = res["similarity_scores"][doc_idx]
                    
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <b>Rank {doc_idx+1}: {meta['doc_name']}</b><br>
                            <small>Page: {meta['page_number']} | ID: {meta['chunk_id'][:10]}</small><br>
                            <small class="badge-similarity">Similarity: {score:.1%}</small><br>
                            <p style="font-size:0.85rem; margin-top:5px;">{doc.page_content[:120]}...</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

# ----------------- PAGE 3: ANALYTICS -----------------
with tab_analytics:
    st.subheader("📊 System & Retrieval Analytics")
    
    # Try to load database stats
    try:
        db = rag_backend.get_db()
        db_data = db.get()
        total_chunks = len(db_data["documents"]) if db_data and "documents" in db_data else 0
        
        # Group chunks by source PDF/txt file
        sources = set()
        chunk_lengths = []
        source_counts = defaultdict(int)
        
        if db_data and "metadatas" in db_data and db_data["metadatas"]:
            for meta, doc_text in zip(db_data["metadatas"], db_data["documents"]):
                s = os.path.basename(meta.get("source", "Unknown"))
                sources.add(s)
                source_counts[s] += 1
                chunk_lengths.append(len(doc_text))
                
        total_pdfs = len(sources)
        avg_chunk_len = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0.0
        
    except Exception as e:
        total_chunks = 0
        total_pdfs = 0
        avg_chunk_len = 0.0
        source_counts = {}
        st.warning(f"Failed to load vector database metrics: {str(e)}")
        
    # Standard metrics display
    col_an1, col_an2, col_an3, col_an4 = st.columns(4)
    col_an1.metric("Total PDFs Ingested", total_pdfs)
    col_an2.metric("Total Chunks Created", total_chunks)
    col_an3.metric("Avg Chunk Character Length", f"{avg_chunk_len:.1f}")
    col_an4.metric("LLM Used", st.session_state.llm_model)
    
    col_an5, col_an6, col_an7 = st.columns(3)
    col_an5.metric("Embedding Model", st.session_state.embedding_model_name)
    
    # Calculate historical query stats
    history_df = pd.DataFrame(st.session_state.query_stats_history)
    if not history_df.empty:
        col_an6.metric("Average Query Time", f"{history_df['retrieval_time'].mean():.3f} s")
        col_an7.metric("Average Similarity Score", f"{history_df['avg_similarity'].mean():.2%}")
    else:
        col_an6.metric("Average Query Time", "N/A")
        col_an7.metric("Average Similarity Score", "N/A")
        
    # Render Interactive Plotly Charts
    st.markdown("### 📈 Visualizations")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Chart 1: Chunk Distribution per Document
        if source_counts:
            df_dist = pd.DataFrame([{"Source": s, "Chunks": c} for s, c in source_counts.items()])
            fig_pie = px.pie(df_dist, names="Source", values="Chunks", title="Chunk Distribution per Source Document", hole=0.3)
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No chunk distribution data available.")
            
    with chart_col2:
        # Chart 2: Historical Latency comparison
        if not history_df.empty:
            fig_lat = px.bar(
                history_df, 
                x="method", 
                y="retrieval_time", 
                color="method", 
                title="Query Retrieval Latency per Method (s)", 
                labels={"retrieval_time": "Latency (s)", "method": "Retrieval Method"}
            )
            fig_lat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
            st.plotly_chart(fig_lat, use_container_width=True)
        else:
            # Placeholder chart demonstrating sample stats
            methods_demo = ["Vector Search", "Hybrid"]
            latencies_demo = [0.08, 0.12]
            df_demo = pd.DataFrame({"Method": methods_demo, "Latency (s)": latencies_demo})
            fig_demo = px.bar(df_demo, x="Method", y="Latency (s)", color="Method", title="Typical Retrieval Latency Benchmark (s)")
            fig_demo.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
            st.plotly_chart(fig_demo, use_container_width=True)

    # Histogram of similarity scores for last query
    if st.session_state.last_results:
        st.markdown("#### Similarity Score Distribution (Last Query)")
        df_hist = pd.DataFrame({"Scores": st.session_state.last_results["similarity_scores"]})
        fig_hist = px.histogram(df_hist, x="Scores", nbins=5, title="Standardized Similarity Scores Histogram", labels={"Scores": "Similarity Score"})
        fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
        st.plotly_chart(fig_hist, use_container_width=True)

# ----------------- PAGE 4: PIPELINE WORKFLOW -----------------
with tab_pipeline:
    st.subheader("🛤️ RAG Pipeline Workflow State")
    st.markdown("Visualization of execution path highlighting stages in real-time:")
    
    # Render pipeline flowchart based on current state
    stages = [
        ("Doc Loading", "Reading PDFs & extracting text from docs/ directory"),
        ("Chunking", "Splitting pages into text blocks using RecursiveCharacterTextSplitter"),
        ("Embedding", "Generating high-dimensional semantic vectors via OpenAI Embeddings"),
        ("Vector DB", "Persisting vectors into Chroma database for local caching"),
        ("Query Rewrite", "Standalone rephrasing of query based on session conversational memory"),
        ("Retriever", "Retrieval execution (Vector Search, Hybrid)"),
        ("LLM Generative", "Contextual query answer generation via ChatOpenAI"),
        ("Final Answer", "Render final output answers on Streamlit frontend")
    ]
    
    current_stage = st.session_state.pipeline_stage
    
    st.markdown('<div class="pipeline-container">', unsafe_allow_html=True)
    for idx, (name, desc) in enumerate(stages):
        is_active = (current_stage == "History Query Rewrite" and name == "Query Rewrite") or \
                    (current_stage == "Retrieving Documents" and name == "Retriever") or \
                    (current_stage == "LLM Answering" and name == "LLM Generative")
                    
        active_class = "active" if is_active else ""
        
        st.markdown(f'<div class="pipeline-node {active_class}"><b>{name}</b><br><small style="font-size:0.75rem; font-weight:normal; opacity:0.8;">{desc}</small></div>', unsafe_allow_html=True)
        if idx < len(stages) - 1:
            st.markdown(f'<div class="pipeline-arrow {active_class}">↓</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------- PAGE 5: EXECUTION LOGS -----------------
with tab_logs:
    st.subheader("📝 Execution & Backend Logs")
    
    # Display full list of execution events captured
    if not rag_backend.execution_logs:
        st.info("No logs captured yet. Execute a search or rebuild database to populate logs.")
    else:
        for idx, event in enumerate(reversed(rag_backend.execution_logs)):
            with st.expander(f"⏱️ [{event['timestamp']}] {event['stage']} - Latency: {event['duration']:.3f} s"):
                st.write(event["details"])
