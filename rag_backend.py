import os
import sys
import time

workspace_dir = os.path.dirname(os.path.abspath(__file__))
if workspace_dir not in sys.path:
    sys.path.append(workspace_dir)
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Generator
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# LangChain Imports
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.retrievers import BM25Retriever

# EnsembleRetriever is available under different LangChain package layouts depending on version.
try:
    from langchain.retrievers import EnsembleRetriever  # type: ignore[attr-defined]
except ImportError:
    try:
        from langchain_classic.retrievers import EnsembleRetriever
    except ImportError:  # pragma: no cover - runtime fallback for legacy installs
        from langchain.retrievers import EnsembleRetriever  # type: ignore[attr-defined]

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_cohere import CohereRerank

# Load environment variables
load_dotenv()

# Logger state
execution_logs = []

def log_event(stage: str, details: Any, start_time: float = None):
    """Log an execution step with timestamp and duration."""
    duration = time.time() - start_time if start_time else 0.0
    execution_logs.append({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage,
        "details": details,
        "duration": duration
    })

# Pydantic model for structured output (Multi-Query / RRF)
class QueryVariations(BaseModel):
    queries: List[str] = Field(description="List of 3 search query variations")

# Dynamic loader for RRF function from Capstone1 files
def load_rrf_function() -> Any:
    """Dynamically reads 11_reciprocal_rank_fusion.py to extract the reciprocal_rank_fusion function.
    This avoids running the top-level script while importing the exact logic."""
    file_path = "11_reciprocal_rank_fusion.py"
    if not os.path.exists(file_path):
        # Fallback implementation if file is missing
        def reciprocal_rank_fusion(chunk_lists, k=60, verbose=True):
            rrf_scores = defaultdict(float)
            all_unique_chunks = {}
            for chunks in chunk_lists:
                for position, chunk in enumerate(chunks, 1):
                    chunk_content = chunk.page_content
                    all_unique_chunks[chunk_content] = chunk
                    rrf_scores[chunk_content] += 1 / (k + position)
            sorted_chunks = sorted(
                [(all_unique_chunks[content], score) for content, score in rrf_scores.items()],
                key=lambda x: x[1],
                reverse=True
            )
            return sorted_chunks
        return reciprocal_rank_fusion
    
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    func_lines = []
    in_func = False
    for line in lines:
        if line.startswith("def reciprocal_rank_fusion"):
            in_func = True
        if in_func:
            func_lines.append(line)
            if "return sorted_chunks" in line:
                break
                
    func_code = "".join(func_lines)
    local_vars = {}
    exec(func_code, globals(), local_vars)
    return local_vars["reciprocal_rank_fusion"]

# Cached initialization of Embeddings
_embeddings = None
def get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key is missing. Please set OPENAI_API_KEY in your .env file.")
        _embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return _embeddings

# Cached initialization of Chroma DB
_db = None
def get_db(persist_directory: str = "db/chroma_db") -> Chroma:
    global _db
    if _db is None:
        embeddings_model = get_embeddings()
        if not os.path.exists(persist_directory):
            raise FileNotFoundError(f"Database directory {persist_directory} not found. Please click 'Build Database' to create it.")
        _db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings_model,
            collection_metadata={"hnsw:space": "cosine"}
        )
    return _db

# Clear cached Chroma DB
def reset_db_cache():
    global _db, _embeddings, _bm25_retriever
    _db = None
    _embeddings = None
    _bm25_retriever = None

# Extract metadata helper
def parse_chunk_metadata(doc: Document, index: int) -> Dict[str, Any]:
    """Extracts clean metadata from a retrieved Document."""
    source_path = doc.metadata.get("source", "Unknown Document")
    doc_name = os.path.basename(source_path)
    
    # Try to find page number (default 0-indexed in LangChain PDF loaders)
    page_num = doc.metadata.get("page", None)
    if page_num is not None:
        # Convert to 1-indexed for user display
        page_display = page_num + 1 if isinstance(page_num, int) else page_num
    else:
        page_display = "N/A"
        
    # Get Chunk ID (prefer Chroma ID if present, otherwise generate a hash or index-based)
    chunk_id = doc.metadata.get("chunk_id", f"Chunk_{index}")
    
    return {
        "doc_name": doc_name,
        "page_number": page_display,
        "chunk_id": chunk_id,
        "source_path": source_path
    }

# Cosine similarity calculator
def get_cosine_similarity(query: str, doc_contents: List[str]) -> List[float]:
    """Computes standardized cosine similarity scores between a query and retrieved document texts."""
    if not doc_contents:
        return []
    embeddings_model = get_embeddings()
    query_emb = np.array(embeddings_model.embed_query(query))
    doc_embs = np.array(embeddings_model.embed_documents(doc_contents))
    
    # Normalize query embedding
    query_norm = np.linalg.norm(query_emb)
    if query_norm == 0:
        return [0.0] * len(doc_contents)
        
    scores = []
    for doc_emb in doc_embs:
        doc_norm = np.linalg.norm(doc_emb)
        if doc_norm == 0:
            scores.append(0.0)
        else:
            similarity = np.dot(query_emb, doc_emb) / (query_norm * doc_norm)
            # Clip between 0 and 1
            scores.append(float(np.clip(similarity, 0.0, 1.0)))
    return scores

# BM25 Retriever
_bm25_retriever = None
_bm25_docs = []
def get_bm25_retriever(k: int) -> Tuple[BM25Retriever, List[Document]]:
    global _bm25_retriever, _bm25_docs
    db = get_db()
    res = db.get()
    
    if not res or not res.get("documents"):
        raise ValueError("Chroma database is empty. Please add source files and click 'Build Database'.")
        
    # Reconstruct document list
    docs = []
    for i, content in enumerate(res["documents"]):
        meta = res["metadatas"][i] if res["metadatas"] else {}
        meta["chunk_id"] = res["ids"][i] if "ids" in res else f"chunk_{i}"
        docs.append(Document(page_content=content, metadata=meta))
        
    _bm25_docs = docs
    _bm25_retriever = BM25Retriever.from_documents(docs)
    _bm25_retriever.k = k
    return _bm25_retriever, docs

# BM25 Scores calculation helper
def get_bm25_scores(retriever: BM25Retriever, corpus: List[Document], query: str, retrieved_docs: List[Document]) -> List[float]:
    """Computes BM25 score for the retrieved documents."""
    try:
        tokens = retriever.preprocess_func(query)
        all_scores = retriever.vectorizer.get_scores(tokens)
        
        # Map page content to BM25 score
        scores_map = {}
        for doc, score in zip(corpus, all_scores):
            scores_map[doc.page_content] = float(score)
            
        retrieved_scores = []
        for r_doc in retrieved_docs:
            retrieved_scores.append(scores_map.get(r_doc.page_content, 0.0))
        return retrieved_scores
    except Exception as e:
        return [0.0] * len(retrieved_docs)

# Standard Retrieval Pipeline interface
def perform_retrieval(query: str, method: str, k: int) -> Dict[str, Any]:
    """Performs document retrieval using the selected method and computes metrics."""
    start_time = time.time()
    db = get_db()
    
    retrieved_docs = []
    scores = []
    raw_scores = []
    generated_queries = []
    
    # 1. Vector Search
    if method == "Vector Search":
        retriever = db.as_retriever(search_kwargs={"k": k})
        retrieved_docs = retriever.invoke(query)
        scores = get_cosine_similarity(query, [d.page_content for d in retrieved_docs])
        raw_scores = [float(s) for s in scores]
        
    # 2. BM25 Search
    elif method == "BM25":
        bm25, corpus = get_bm25_retriever(k)
        retrieved_docs = bm25.invoke(query)
        scores = get_cosine_similarity(query, [d.page_content for d in retrieved_docs])
        raw_scores = get_bm25_scores(bm25, corpus, query, retrieved_docs)
        
    # 3. Hybrid
    elif method == "Hybrid":
        vector_ret = db.as_retriever(search_kwargs={"k": k})
        bm25, corpus = get_bm25_retriever(k)
        ensemble = EnsembleRetriever(retrievers=[vector_ret, bm25], weights=[0.7, 0.3])
        retrieved_docs = ensemble.invoke(query)
        scores = get_cosine_similarity(query, [d.page_content for d in retrieved_docs])
        raw_scores = [float(s) for s in scores] # Hybrid doesn't give scores directly, use cosine
        
    # 4. Multi Query
    elif method == "Multi Query":
        # Generate variations using LLM
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        llm_with_tools = llm.with_structured_output(QueryVariations)
        
        prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents:
Original query: {query}
Return 3 alternative queries that rephrase or approach the same question from different angles."""
        
        response = llm_with_tools.invoke(prompt)
        query_variations = response.queries
        generated_queries = query_variations
        
        vector_ret = db.as_retriever(search_kwargs={"k": k})
        unique_docs = {}
        for var in query_variations:
            docs = vector_ret.invoke(var)
            for d in docs:
                unique_docs[d.page_content] = d
        retrieved_docs = list(unique_docs.values())[:k]
        scores = get_cosine_similarity(query, [d.page_content for d in retrieved_docs])
        raw_scores = [float(s) for s in scores]
        
    # 5. RRF (Reciprocal Rank Fusion)
    elif method == "RRF":
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        llm_with_tools = llm.with_structured_output(QueryVariations)
        
        prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents:
Original query: {query}
Return 3 alternative queries that rephrase or approach the same question from different angles."""
        
        response = llm_with_tools.invoke(prompt)
        query_variations = response.queries
        generated_queries = query_variations
        
        vector_ret = db.as_retriever(search_kwargs={"k": k})
        all_retrievals = []
        for var in query_variations:
            docs = vector_ret.invoke(var)
            all_retrievals.append(docs)
            
        rrf_func = load_rrf_function()
        fused = rrf_func(all_retrievals, k=60, verbose=False)
        
        retrieved_docs = [doc for doc, _ in fused[:k]]
        raw_scores = [float(score) for _, score in fused[:k]]
        scores = get_cosine_similarity(query, [d.page_content for d in retrieved_docs])
        
    # 6. Reranker (Cohere)
    elif method == "Reranker":
        if not os.environ.get("COHERE_API_KEY") and not os.environ.get("CO_API_KEY"):
            raise ValueError("Cohere API key is missing. Please set COHERE_API_KEY in your .env file.")
            
        vector_ret = db.as_retriever(search_kwargs={"k": k * 2}) # Retrieve more for reranking
        bm25, corpus = get_bm25_retriever(k * 2)
        ensemble = EnsembleRetriever(retrievers=[vector_ret, bm25], weights=[0.7, 0.3])
        candidate_docs = ensemble.invoke(query)
        
        # Run Cohere Rerank
        cohere_key = os.environ.get("COHERE_API_KEY") or os.environ.get("CO_API_KEY")
        reranker = CohereRerank(model="rerank-english-v3.0", top_n=k, cohere_api_key=cohere_key)
        retrieved_docs = reranker.compress_documents(candidate_docs, query)
        
        scores = get_cosine_similarity(query, [d.page_content for d in retrieved_docs])
        # Cohere reranker score is in metadata relevance_score
        raw_scores = [float(d.metadata.get("relevance_score", 0.0)) for d in retrieved_docs]

    duration = time.time() - start_time
    
    # Log the event
    log_event(
        stage=f"Retrieval ({method})",
        details={
            "query": query,
            "k": k,
            "docs_found": len(retrieved_docs),
            "generated_queries": generated_queries,
            "latency": duration
        },
        start_time=start_time
    )
    
    return {
        "documents": retrieved_docs,
        "similarity_scores": scores,
        "raw_scores": raw_scores,
        "retrieval_time": duration,
        "generated_queries": generated_queries
    }

# Conversational Memory Standalone Query re-writer (reusing 4_history_aware_generation.py logic)
def rewrite_query_with_history(query: str, chat_history: List[Tuple[str, str]], llm_model: str = "gpt-4o") -> str:
    """Rewrites a user's question to be standalone and searchable, based on chat history."""
    if not chat_history:
        return query
        
    start_time = time.time()
    model = ChatOpenAI(model=llm_model, temperature=0)
    
    # Construct history messages
    history_messages = []
    for user_msg, ai_msg in chat_history:
        history_messages.append(HumanMessage(content=user_msg))
        history_messages.append(AIMessage(content=ai_msg))
        
    messages = [
        SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question without any prefix or quotes."),
    ] + history_messages + [
        HumanMessage(content=f"New question: {query}")
    ]
    
    result = model.invoke(messages)
    rewritten = result.content.strip()
    
    log_event(
        stage="History-Aware Query Rewrite",
        details={
            "original_query": query,
            "rewritten_query": rewritten,
            "history_len": len(chat_history)
        },
        start_time=start_time
    )
    return rewritten

# Stream Answer Generation
def generate_rag_answer_stream(query: str, retrieved_docs: List[Document], chat_history: List[Tuple[str, str]], temperature: float, llm_model: str = "gpt-4o") -> Generator[str, None, None]:
    """Generates a streamed answer for the user query using the retrieved context documents."""
    start_time = time.time()

    if not retrieved_docs:
        fallback = "I don't have enough information to answer that question based on the provided documents."
        log_event(
            stage="LLM Answer Generation",
            details={
                "query": query,
                "fallback_used": True,
                "docs_count": 0
            },
            start_time=start_time
        )
        yield fallback
        return

    # Context format identical to 3_answer_generation.py
    context_str = "\n".join([f"- {doc.page_content}" for doc in retrieved_docs])

    combined_input = f"""Answer the question using only the relevant information in the retrieved document excerpts.

Important rules:
1. Use the excerpts as the primary source of truth.
2. If the excerpts contain relevant information, answer from them even if the wording is not an exact match.
3. Only say "I don't have enough information to answer that question based on the provided documents." when the retrieved excerpts do not contain any relevant evidence for the question.
4. If the context is partial, provide the best answer supported by the available evidence and clearly label it as based on the provided documents.

Question: {query}

Retrieved excerpts:
{context_str}
"""

    # Message building
    history_messages = []
    for user_msg, ai_msg in chat_history:
        history_messages.append(HumanMessage(content=user_msg))
        history_messages.append(AIMessage(content=ai_msg))

    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on the provided documents and conversation history. If the documents contain relevant evidence, answer from that evidence. Only refuse when the documents truly lack the needed information."),
    ] + history_messages + [
        HumanMessage(content=combined_input)
    ]

    model = ChatOpenAI(model=llm_model, temperature=temperature, streaming=True)

    full_response = ""
    for chunk in model.stream(messages):
        content = chunk.content
        full_response += content
        yield content

    log_event(
        stage="LLM Answer Generation",
        details={
            "prompt_length_chars": len(combined_input),
            "response": full_response,
            "docs_count": len(retrieved_docs)
        },
        start_time=start_time
    )

# PDF converter
def save_uploaded_pdf(uploaded_file) -> str:
    """Saves uploaded PDF and extracts its text to .txt inside docs/ directory."""
    start_time = time.time()
    os.makedirs("docs", exist_ok=True)
    
    # Save raw pdf
    pdf_path = os.path.join("docs", uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    # Convert PDF to TXT
    txt_name = os.path.splitext(uploaded_file.name)[0] + ".txt"
    txt_path = os.path.join("docs", txt_name)
    
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += f"\n--- Page {i+1} ---\n"
            text += page_text + "\n"
            
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text.strip())
        
    log_event(
        stage="Document Ingestion Loader",
        details={
            "pdf_file": pdf_path,
            "txt_file": txt_path,
            "extracted_chars": len(text)
        },
        start_time=start_time
    )
    return txt_path

# Run DB Ingestion
def run_db_ingestion() -> str:
    """Runs the ingestion pipeline script directly to rebuild the database."""
    start_time = time.time()
    import ingestion_pipeline
    
    log_event("Database Construction Inception", "Starting ingestion_pipeline.py rebuild...")
    
    # Re-run ingestion functions
    documents = ingestion_pipeline.load_documents(docs_path="docs")
    chunks = ingestion_pipeline.split_documents(documents)
    
    # Save to db/chroma_db
    vectorstore = ingestion_pipeline.create_vector_store(chunks, persist_directory="db/chroma_db")
    
    # Refresh active cached instance
    reset_db_cache()
    
    duration = time.time() - start_time
    log_event(
        stage="Database Construction Complete",
        details={
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "persistence_dir": "db/chroma_db"
        },
        start_time=start_time
    )
    
    return f"Success! Loaded {len(documents)} document(s) and split into {len(chunks)} chunks."
