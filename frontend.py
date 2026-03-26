import os
import time
import requests
import streamlit as st

# Config
API_URL = "http://localhost:8000/api"

st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto"  # Automatically collapses sidebar on mobile
)

# Custom Styling (Dark/Modern & Responsive)
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
    }
    
    div.stChatFloatingInputContainer {
        padding-bottom: 2rem;
    }
    
    /* Customize the file uploader */
    div[data-testid="stFileUploader"] {
        padding: 10px;
        border-radius: 12px;
        border: 1px dotted #3d4b60;
    }
    
    /* Title modern look */
    .title-text {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        color: #f1f2f6;
        font-size: 2.5rem;
    }

    /* Mobile Responsive CSS */
    @media screen and (max-width: 768px) {
        .title-text {
            font-size: 1.8rem;
            text-align: center;
        }
        div.stChatFloatingInputContainer {
            padding-bottom: 0.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Title Area
st.markdown("<h1 class='title-text'>🤖 Intelligent Document Q&A</h1>", unsafe_allow_html=True)
st.markdown("Upload your documents securely and ask intelligent queries using state-of-the-art AI.")
st.divider()

# Session State for Conversation and Engine
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("📂 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Upload new files into the knowledge base", 
        type=["pdf", "txt", "csv", "docx", "xlsx", "json", "md"],
        accept_multiple_files=True
    )
    
    if st.button("Process & Rebuild Index", type="primary", use_container_width=True):
        if uploaded_files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                progress_bar.progress(0.1)
                status_text.text("Sending files to backend and rebuilding Vector Knowledge Base... This may take a while.")
                
                # Send files via multi-part form data
                files_payload = [
                    ("files", (uf.name, uf.getvalue(), uf.type)) for uf in uploaded_files
                ]
                start_time = time.time()
                response = requests.post(f"{API_URL}/ingest", files=files_payload)
                response.raise_for_status()
                
                progress_bar.progress(1.0)
                status_text.text(f"Indexing complete in {time.time() - start_time:.1f}s!")
                st.success(response.json().get("message", "Indexing completed successfully!"))
            except Exception as e:
                status_text.text("")
                st.error(f"Error communicating with backend API: {e}")
        else:
            st.warning("Please upload at least one file to re-index.")

    st.divider()
    
    st.markdown("### 📚 Supported Files Loaded")
    try:
        response = requests.get(f"{API_URL}/files")
        if response.status_code == 200:
            filtered_files = response.json().get("files", [])
            if not filtered_files:
                st.info("No supported documents currently loaded.")
            else:
                for f in filtered_files:
                    cols = st.columns([0.85, 0.15])
                    with cols[0]:
                        st.markdown(f"`{f}`")
                    with cols[1]:
                        if st.button("🗑️", key=f"del_{f}", help="Delete this file"):
                            with st.spinner(""):
                                try:
                                    del_resp = requests.delete(f"{API_URL}/files/{f}")
                                    if del_resp.status_code == 200:
                                        st.toast(del_resp.json().get("message", "Deleted successfully"))
                                        # Use st.rerun() if available, otherwise st.experimental_rerun()
                                        if hasattr(st, "rerun"):
                                            st.rerun()
                                        else:
                                            st.experimental_rerun()
                                    else:
                                        st.error(f"Failed to delete: HTTP {del_resp.status_code}")
                                except Exception as e:
                                    st.error(f"Error: {e}")
        else:
            st.error(f"Backend API error: HTTP {response.status_code}")
    except requests.exceptions.ConnectionError:
            st.error("Backend API is not reachable. Is FastAPI running on port 8000?")
    except Exception as e:
        st.error(f"Could not read files from backend: {e}")

# Display previously typed chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "context" in message and message["context"]:
            with st.expander("🔍 View Retrieved Context (Transparency & Explainability)"):
                st.markdown("**The following document chunks were retrieved and used to generate this answer:**")
                for i, chunk in enumerate(message["context"]):
                    st.info(f"**Chunk {i+1}:**\n\n{chunk}")

# Chat input container
if query := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Get answer
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            try:
                # Supply recent message history (excluding current user query)
                recent_history = [m for m in st.session_state.messages[:-1] if m.get("role") in ["user", "assistant"]]
                payload = {
                    "query": query,
                    "chat_history": recent_history
                }
                
                start_time = time.time()
                api_response = requests.post(f"{API_URL}/query", json=payload)
                api_response.raise_for_status()
                result = api_response.json()
                
                response_text = result.get("response", "")
                context_chunks = result.get("context", [])
                
                message_placeholder.markdown(response_text)
                
                # Render transparency context drawer directly
                if context_chunks:
                    with st.expander("🔍 View Retrieved Context (Transparency & Explainability)"):
                        st.markdown("**The following document chunks were retrieved and used to generate this answer:**")
                        for i, chunk in enumerate(context_chunks):
                            st.info(f"**Chunk {i+1}:**\n\n{chunk}")

                # Keep history
                st.session_state.messages.append({"role": "assistant", "content": response_text, "context": context_chunks})
            except requests.exceptions.ConnectionError:
                error_msg = "Backend API is not reachable. Please ensure FastAPI is running."
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except Exception as e:
                error_msg = f"Error during search: {e}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
