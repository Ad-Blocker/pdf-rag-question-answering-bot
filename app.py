import streamlit as st
from pypdf import PdfReader
import os
import time
import google.generativeai as genai
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Gemini PDF QA Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium Modern Dark UI Styles ---
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');

    /* Global styling overrides */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Sidebar container styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #334155;
    }

    /* Card styling (Glassmorphism effect) */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* Status indicators styling */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 6px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 12px;
    }
    .status-active {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-inactive {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .status-idle {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .status-progress {
        background-color: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.6);
        border: 2px dashed rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 20px;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(99, 102, 241, 0.8);
        background: rgba(15, 23, 42, 0.8);
    }

    /* Custom chat bubbles */
    .chat-bubble {
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 12px;
        max-width: 90%;
        line-height: 1.5;
    }
    .chat-bubble.assistant {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #818cf8;
        align-self: flex-start;
        margin-right: auto;
    }
    .chat-bubble.user {
        background: linear-gradient(135deg, #4f46e5 0%, #3730a3 100%);
        align-self: flex-end;
        margin-left: auto;
        color: #ffffff;
    }
    
    /* Code block styling in expanders */
    .debug-code {
        background-color: #0f172a;
        color: #38bdf8;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #1e293b;
        font-family: 'Courier New', Courier, monospace;
        font-size: 0.9rem;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "pdf_data" not in st.session_state:
    st.session_state.pdf_data = {
        "filename": None,
        "raw_text": "",
        "pages": [],
        "char_count": 0,
        "processed": False,
        "chunks": [],
        "embeddings": [],
        "indexed": False,
        "embedding_model": "models/text-embedding-004",
        "embedding_dim": None,
        "indexing_error": None,
        "current_query": None,
        "retrieved_chunks": [],
        "chat_history": [],
        "qa_model": "models/gemini-1.5-flash"
    }

# --- Chunker Logic ---
def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Splits text into chunks of specified characters with overlap.
    Handles small documents gracefully.
    """
    chunks = []
    text = text.strip()
    if not text:
        return []
    
    if len(text) <= chunk_size:
        return [text]
        
    step = chunk_size - overlap
    if step <= 0:
        step = 1  # Safety check to prevent infinite loop
        
    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if i + chunk_size >= len(text):
            break
    return chunks

# --- Embedding Generation Logic ---
def generate_embeddings(chunks, api_key, model="models/text-embedding-004"):
    """
    Calls Google Generative AI API to retrieve embeddings for each chunk in batches.
    """
    genai.configure(api_key=api_key)
    
    # Google embed_content supports batching
    batch_size = 100
    embeddings = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        response = genai.embed_content(
            model=model,
            content=batch,
            task_type="retrieval_document"
        )
        embeddings.extend(response['embedding'])
        
    return embeddings

# --- Cosine Similarity and Retrieval Logic ---
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

def retrieve_top_k(query, chunks, chunk_embeddings, api_key, model, top_k=3):
    genai.configure(api_key=api_key)
    
    # 1. Generate query embedding
    response = genai.embed_content(
        model=model,
        content=query,
        task_type="retrieval_query"
    )
    query_embedding = np.array(response['embedding'])
    
    # 2. Compute similarity for all chunk embeddings
    results = []
    for idx, chunk_emb in enumerate(chunk_embeddings):
        emb_arr = np.array(chunk_emb)
        sim = cosine_similarity(query_embedding, emb_arr)
        results.append((idx, sim))
        
    # 3. Sort by similarity descending
    results.sort(key=lambda x: x[1], reverse=True)
    top_results = results[:top_k]
    
    # 4. Package results
    retrieved = []
    for idx, sim in top_results:
        retrieved.append({
            "chunk_idx": idx,
            "text": chunks[idx],
            "similarity": sim
        })
        
    return retrieved

# --- Answer Generation Logic ---
def generate_answer(query, retrieved_chunks, api_key, model_name="gemini-1.5-flash"):
    """
    Constructs a grounded RAG prompt and gets response from Gemini API.
    """
    genai.configure(api_key=api_key)
    
    # 1. Build the context string
    context_str = ""
    for item in retrieved_chunks:
        chunk_num = item["chunk_idx"] + 1
        context_str += f"[Context Chunk #{chunk_num}]:\n{item['text']}\n\n"
        
    # 2. Build system instruction
    system_instruction = (
        "You are a helpful, precise question-answering assistant. You must answer the user's question "
        "based ONLY on the provided Context. Do not make up facts or use outside knowledge. "
        "If the information required to answer the question is not present in the Context, "
        "state clearly and explicitly that the information is not available in the document. "
        "Always be honest and direct about what is and isn't supported by the context."
    )
    
    # 3. Setup prompt
    model = genai.GenerativeModel(model_name)
    prompt = f"{system_instruction}\n\nContext:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
    
    # 4. Generate answer
    response = model.generate_content(prompt)
    return response.text

# --- Sidebar Content ---
with st.sidebar:
    st.markdown("### 🌌 ANTIGRAVITY | PDF Bot")
    st.caption("Phase 4: Complete Grounded QA MVP")
    st.markdown("---")

    # API Key Input
    st.markdown("#### 🔑 Gemini API Credentials")
    env_api_key = os.getenv("GEMINI_API_KEY", "")
    if env_api_key:
        st.success("API Key loaded from environment variables.")
        api_key = env_api_key
    else:
        api_key = st.text_input(
            "Enter Gemini API Key",
            type="password",
            placeholder="AIzaSy...",
            help="Get your API key from Google AI Studio"
        )
        if not api_key:
            st.warning("Please enter your API key to enable embedding & QA generation.")

    st.markdown("---")
    st.markdown("#### ⚙️ Embedding Settings")
    
    # Standard models list to fall back on
    fallback_models = ["models/text-embedding-004", "models/embedding-001"]
    available_models = fallback_models
    
    # Try listing models dynamically if the API key is entered
    if api_key:
        try:
            genai.configure(api_key=api_key)
            fetched = []
            for m in genai.list_models():
                if 'embedContent' in m.supported_generation_methods:
                    fetched.append(m.name)
            if fetched:
                available_models = fetched
        except Exception:
            pass
            
    # Ensure the current session model is present in the list
    current_model = st.session_state.pdf_data["embedding_model"]
    if current_model not in available_models:
        available_models = [current_model] + [m for m in available_models if m != current_model]
        
    selected_model = st.selectbox(
        "Select Embedding Model",
        options=available_models,
        index=available_models.index(current_model) if current_model in available_models else 0,
        help="If one model fails (e.g. 404), try switching to another model like models/embedding-001."
    )
    
    # If the user switches model, reset the indexed state so they re-generate embeddings
    if selected_model != current_model:
        st.session_state.pdf_data["embedding_model"] = selected_model
        st.session_state.pdf_data["embeddings"] = []
        st.session_state.pdf_data["indexed"] = False
        st.session_state.pdf_data["embedding_dim"] = None
        st.session_state.pdf_data["indexing_error"] = None
        st.rerun()

    st.markdown("---")
    st.markdown("#### 💬 QA Generation Settings")
    
    fallback_gen_models = ["models/gemini-1.5-flash", "models/gemini-1.5-pro", "models/gemini-2.5-flash"]
    available_gen_models = fallback_gen_models
    
    if api_key:
        try:
            genai.configure(api_key=api_key)
            fetched = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    fetched.append(m.name)
            if fetched:
                available_gen_models = fetched
        except Exception:
            pass
            
    current_qa_model = st.session_state.pdf_data.get("qa_model", "models/gemini-1.5-flash")
    if current_qa_model not in available_gen_models:
        available_gen_models = [current_qa_model] + [m for m in available_gen_models if m != current_qa_model]
        
    selected_qa_model = st.selectbox(
        "Select QA Model",
        options=available_gen_models,
        index=available_gen_models.index(current_qa_model) if current_qa_model in available_gen_models else 0,
        help="Choose the Gemini model used to generate answers. Switch models if a 404 error occurs."
    )
    
    if selected_qa_model != current_qa_model:
        st.session_state.pdf_data["qa_model"] = selected_qa_model
        st.rerun()

    st.markdown("---")
    st.markdown("#### 📊 Document Status")
    
    if st.session_state.pdf_data["processed"]:
        st.write(f"**Filename:** {st.session_state.pdf_data['filename']}")
        st.write(f"**Total Pages:** {len(st.session_state.pdf_data['pages'])}")
        st.write(f"**Characters:** {st.session_state.pdf_data['char_count']:,}")
    else:
        st.markdown('<div class="status-badge status-inactive">🔴 No Document Uploaded</div>', unsafe_allow_html=True)

    st.markdown("---")
    # Reset button
    if st.button("🧹 Clear Session & Cache"):
        st.session_state.pdf_data = {
            "filename": None,
            "raw_text": "",
            "pages": [],
            "char_count": 0,
            "processed": False,
            "chunks": [],
            "embeddings": [],
            "indexed": False,
            "embedding_model": "models/text-embedding-004",
            "embedding_dim": None,
            "indexing_error": None,
            "current_query": None,
            "retrieved_chunks": [],
            "chat_history": [],
            "qa_model": "models/gemini-1.5-flash"
        }
        st.rerun()

# --- Main Page Layout ---
st.title("🤖 PDF Question Answering Bot")
st.markdown("#### Upload a PDF, index its content, and chat with it locally using Gemini AI.")

# Layout Columns
col_upload, col_chat = st.columns([1, 1.2])

# Left Column: Upload & Ingestion
with col_upload:
    # 1. File Upload Card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📄 Upload Document")
    
    uploaded_file = st.file_uploader("Select a PDF file", type=["pdf"])
    
    # Process file if uploaded and not already processed
    if uploaded_file is not None:
        if st.session_state.pdf_data["filename"] != uploaded_file.name:
            with st.spinner("⏳ Extracting text and chunking PDF..."):
                try:
                    reader = PdfReader(uploaded_file)
                    raw_text_list = []
                    pages_info = []
                    
                    # Page text extraction
                    for idx, page in enumerate(reader.pages):
                        page_text = page.extract_text() or ""
                        raw_text_list.append(page_text)
                        pages_info.append({
                            "page_num": idx + 1,
                            "char_count": len(page_text),
                            "text": page_text
                        })
                    
                    full_text = "\n\n".join(raw_text_list)
                    
                    # Error Handling: Empty PDF
                    if not full_text.strip():
                        raise ValueError("The uploaded PDF contains no extractable text. It might be scan-only (image-only) or empty.")
                    
                    # Chunking text
                    chunks = chunk_text(full_text, chunk_size=1000, overlap=200)
                    
                    # Error Handling: Document too small
                    if not chunks:
                        raise ValueError("The PDF document does not contain enough characters to index.")
                    
                    # Save to state
                    st.session_state.pdf_data = {
                        "filename": uploaded_file.name,
                        "raw_text": full_text,
                        "pages": pages_info,
                        "char_count": len(full_text),
                        "processed": True,
                        "chunks": chunks,
                        "embeddings": [],
                        "indexed": False,
                        "embedding_model": "models/text-embedding-004",
                        "embedding_dim": None,
                        "indexing_error": None,
                        "current_query": None,
                        "retrieved_chunks": [],
                        "chat_history": [],
                        "qa_model": "models/gemini-1.5-flash"
                    }
                    st.success("✅ PDF extracted and chunked successfully!")
                    st.rerun()
                except Exception as e:
                    st.session_state.pdf_data = {
                        "filename": uploaded_file.name,
                        "raw_text": "",
                        "pages": [],
                        "char_count": 0,
                        "processed": False,
                        "chunks": [],
                        "embeddings": [],
                        "indexed": False,
                        "embedding_model": "models/text-embedding-004",
                        "embedding_dim": None,
                        "indexing_error": str(e),
                        "current_query": None,
                        "retrieved_chunks": [],
                        "chat_history": [],
                        "qa_model": "models/gemini-1.5-flash"
                    }
                    st.error(f"❌ Processing failed: {str(e)}")
                    st.rerun()
                    
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. Embedding Index Generation (automatic when API key and processed PDF are present)
    if st.session_state.pdf_data["processed"] and not st.session_state.pdf_data["indexed"]:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("⚡ Embeddings Generation")
        
        if not api_key:
            st.warning("⚠️ Please enter a Gemini API Key in the sidebar to start generating embeddings.")
        else:
            if st.session_state.pdf_data["indexing_error"]:
                st.error(f"❌ Previous indexing attempt failed: {st.session_state.pdf_data['indexing_error']}")
                if st.button("🔄 Retry Embedding Generation"):
                    st.session_state.pdf_data["indexing_error"] = None
                    st.rerun()
            else:
                st.info(f"Ready to generate embeddings for {len(st.session_state.pdf_data['chunks'])} chunks.")
                if st.button("🚀 Start Indexing"):
                    selected_model = st.session_state.pdf_data["embedding_model"]
                    with st.spinner(f"⏳ Generating embeddings using {selected_model}..."):
                        try:
                            embeddings = generate_embeddings(
                                st.session_state.pdf_data["chunks"], 
                                api_key, 
                                model=selected_model
                            )
                            
                            # Save to state
                            st.session_state.pdf_data["embeddings"] = embeddings
                            st.session_state.pdf_data["embedding_dim"] = len(embeddings[0]) if embeddings else None
                            st.session_state.pdf_data["indexed"] = True
                            st.session_state.pdf_data["indexing_error"] = None
                            st.success("🎉 Embeddings generated and document indexed!")
                            st.rerun()
                        except Exception as e:
                            st.session_state.pdf_data["indexing_error"] = str(e)
                            st.error(f"❌ Embedding API Failure: {str(e)}")
                            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. Index Statistics Card
    if st.session_state.pdf_data["processed"]:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("⚙️ Index Statistics")
        
        # Setup stats status
        if st.session_state.pdf_data["indexed"]:
            status_badge = '<span class="status-badge status-active">🟢 Ready for Questions</span>'
        elif st.session_state.pdf_data["indexing_error"]:
            status_badge = '<span class="status-badge status-inactive">🔴 Indexing Failed</span>'
        else:
            status_badge = '<span class="status-badge status-idle">🟡 Pending Embeddings</span>'
            
        st.markdown(f"**Index Status:** {status_badge}", unsafe_allow_html=True)
        
        # Display grid statistics
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Total Chunks", len(st.session_state.pdf_data["chunks"]))
            st.metric("Chunk Size", "1,000 chars")
            st.metric("Overlap Size", "200 chars")
        with col_s2:
            st.metric("Embedding Dim", st.session_state.pdf_data["embedding_dim"] or "N/A")
            # Extract just the model name for a cleaner metric UI
            model_display_name = st.session_state.pdf_data["embedding_model"].split('/')[-1]
            st.metric("Embedding Model", model_display_name)
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 4. Debug View Expander
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        with st.expander("🛠️ Index Debug View"):
            st.write(f"**Number of Embeddings Generated:** {len(st.session_state.pdf_data['embeddings'])}")
            
            if st.session_state.pdf_data["chunks"]:
                st.markdown("**First Chunk (Chunk #1):**")
                st.markdown(f'<div class="debug-code">{st.session_state.pdf_data["chunks"][0]}</div>', unsafe_allow_html=True)
                
                if len(st.session_state.pdf_data["chunks"]) > 1:
                    st.markdown("---")
                    st.markdown(f"**Last Chunk (Chunk #{len(st.session_state.pdf_data['chunks'])}):**")
                    st.markdown(f'<div class="debug-code">{st.session_state.pdf_data["chunks"][-1]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Right Column: Chat QA Interface
with col_chat:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("💬 Interactive Chat Assistant")
    
    # Handle chat visual states
    if not st.session_state.pdf_data["processed"]:
        st.info("💡 Once a PDF document is uploaded and indexed, you can ask questions about its content here.")
        st.markdown(
            '<div class="chat-bubble assistant">👋 Welcome! Please upload a PDF file on the left to get started.</div>', 
            unsafe_allow_html=True
        )
    elif not st.session_state.pdf_data["indexed"]:
        st.warning("⚠️ Embedding indexing is pending. Provide your Gemini API key and click 'Start Indexing' to begin.")
        st.markdown(
            f'<div class="chat-bubble assistant">📚 I have chunked the document <b>{st.session_state.pdf_data["filename"]}</b> '
            f'into {len(st.session_state.pdf_data["chunks"])} passages.<br><br>Please enter your API Key and start the indexing phase '
            f'to generate embedding vectors. This is required before we can compute similarity for search.</div>',
            unsafe_allow_html=True
        )
    else:
        # App is indexed and ready for questions
        st.success("🎉 Ready for Questions! Ask anything below.")
        
        # User input form for QA
        with st.form("qa_form", clear_on_submit=True):
            user_query = st.text_input(
                "Ask a question about the PDF:",
                placeholder="e.g., What is the primary methodology used in this document?",
                value=""
            )
            qa_submitted = st.form_submit_button("💬 Ask Gemini")
            
        # Perform retrieval and question answering on submit
        if qa_submitted and user_query.strip():
            with st.spinner("⏳ Retrieving context and generating answer..."):
                try:
                    # 1. Retrieve top 3 relevant chunks
                    retrieved = retrieve_top_k(
                        query=user_query,
                        chunks=st.session_state.pdf_data["chunks"],
                        chunk_embeddings=st.session_state.pdf_data["embeddings"],
                        api_key=api_key,
                        model=st.session_state.pdf_data["embedding_model"],
                        top_k=3
                    )
                    
                    # 2. Generate grounded answer using the selected QA model
                    selected_qa_model = st.session_state.pdf_data["qa_model"]
                    answer = generate_answer(
                        query=user_query,
                        retrieved_chunks=retrieved,
                        api_key=api_key,
                        model_name=selected_qa_model
                    )
                    
                    # 3. Save current query & retrieved context for debug view
                    st.session_state.pdf_data["current_query"] = user_query
                    st.session_state.pdf_data["retrieved_chunks"] = retrieved
                    
                    # 4. Map sources for attribution
                    sources_list = [
                        {"chunk_num": item["chunk_idx"] + 1, "similarity": item["similarity"]} 
                        for item in retrieved
                    ]
                    
                    # 5. Save user & assistant messages to history
                    st.session_state.pdf_data["chat_history"].append({
                        "role": "user",
                        "content": user_query
                    })
                    st.session_state.pdf_data["chat_history"].append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources_list
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Operation failed: {str(e)}")
                    
        # Render Chat History
        if st.session_state.pdf_data["chat_history"]:
            st.markdown("### 💬 Chat History")
            
            for msg in st.session_state.pdf_data["chat_history"]:
                role = msg["role"]
                content = msg["content"]
                sources = msg.get("sources", [])
                
                if role == "user":
                    st.markdown(
                        f'<div class="chat-bubble user"><b>You:</b> {content}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="chat-bubble assistant"><b>Assistant:</b><br>{content}</div>',
                        unsafe_allow_html=True
                    )
                    # Display source attribution directly below the assistant message
                    if sources:
                        source_chips = []
                        for s in sources:
                            source_chips.append(
                                f"<span style='background: rgba(129, 140, 248, 0.12); color: #818cf8; border: 1px solid rgba(129, 140, 248, 0.3); padding: 3px 8px; border-radius: 6px; font-size: 0.82rem; margin-right: 6px;'>"
                                f"Chunk #{s['chunk_num']} (Sim: {s['similarity']:.4f})"
                                f"</span>"
                            )
                        sources_html = "".join(source_chips)
                        st.markdown(
                            f'<div style="margin-top: -6px; margin-bottom: 18px; padding-left: 10px; display: flex; flex-wrap: wrap; align-items: center;">'
                            f'<span style="font-size: 0.85rem; color: #94a3b8; font-weight: 600; margin-right: 8px;">🔍 Sources:</span>'
                            f'{sources_html}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
        else:
            st.markdown(
                f'<div class="chat-bubble assistant">📚 <b>{st.session_state.pdf_data["filename"]}</b> has been successfully indexed '
                f'using <code>{st.session_state.pdf_data["embedding_model"]}</code>.<br><br>'
                f'Ask a question above to retrieve context matching and generate fully grounded answers!</div>',
                unsafe_allow_html=True
            )

        # 5. Maintain Retrieval Debug Panel (shows detailed context text for the most recent question)
        if st.session_state.pdf_data["retrieved_chunks"]:
            st.markdown("---")
            st.markdown("### 🔎 Semantic Retrieval Debug Panel (Most Recent)")
            st.caption(f"Showing top matches for: *\"{st.session_state.pdf_data['current_query']}\"*")
            
            for rank, item in enumerate(st.session_state.pdf_data["retrieved_chunks"]):
                chunk_num = item["chunk_idx"] + 1
                score = item["similarity"]
                text_content = item["text"]
                
                st.markdown(f"""
                <div style="background: rgba(99, 102, 241, 0.04); border: 1px solid rgba(99, 102, 241, 0.15); border-radius: 10px; padding: 16px; margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; font-weight: bold; margin-bottom: 8px;">
                        <span>Rank #{rank + 1} &nbsp;|&nbsp; Chunk #{chunk_num} (Index: {item['chunk_idx']})</span>
                        <span style="color: #818cf8;">Cos Similarity: {score:.4f}</span>
                    </div>
                    <div style="font-size: 0.95rem; line-height: 1.5; color: #cbd5e1; font-family: 'Plus Jakarta Sans', sans-serif;">{text_content}</div>
                </div>
                """, unsafe_allow_html=True)
                
    st.markdown('</div>', unsafe_allow_html=True)
