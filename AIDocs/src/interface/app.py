"""Modern Streamlit interface for the RAG system."""
import streamlit as st
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.main import RAGPipeline
from src.utils import setup_logger, get_paths

logger = setup_logger(__name__)


# Page configuration
st.set_page_config(
    page_title="Context-Aware RAG Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --background-dark: #0f172a;
        --surface-color: #1e293b;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom header */
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.3);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    /* Chat messages */
    .user-message {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 1rem 1rem 0.25rem 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2);
    }
    
    .assistant-message {
        background: #f8fafc;
        color: #0f172a;
        padding: 1rem 1.5rem;
        border-radius: 1rem 1rem 1rem 0.25rem;
        margin: 1rem 0;
        border-left: 4px solid #6366f1;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Citations */
    .citation {
        background: #eff6ff;
        border-left: 3px solid #3b82f6;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        font-size: 0.9rem;
    }
    
    .citation-title {
        font-weight: 600;
        color: #1e40af;
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border: 1px solid #e2e8f0;
        margin: 0.5rem 0;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #6366f1;
    }
    
    .stat-label {
        color: #64748b;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 0.75rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
    }
    
    /* File uploader */
    .uploadedFile {
        border: 2px dashed #6366f1;
        border-radius: 1rem;
        padding: 1rem;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: #f8fafc;
    }
    
    /* Success/Info boxes */
    .stSuccess, .stInfo {
        border-radius: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []


def initialize_pipeline():
    """Initialize the RAG pipeline."""
    if st.session_state.pipeline is None:
        with st.spinner("🚀 Initializing RAG Pipeline..."):
            st.session_state.pipeline = RAGPipeline()
        st.success("✅ Pipeline initialized!")


def process_uploaded_files(uploaded_files):
    """Process uploaded PDF files."""
    if not uploaded_files:
        return
    
    initialize_pipeline()
    paths = get_paths()
    
    # Save uploaded files
    saved_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(paths['raw_pdfs_dir'], uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)
    
    # Process files
    with st.spinner(f"📄 Processing {len(saved_paths)} PDF(s)..."):
        results = st.session_state.pipeline.process_multiple_pdfs(saved_paths)
    
    # Update processed files
    st.session_state.processed_files.extend(results)
    
    return results


def display_answer(result):
    """Display answer with citations."""
    # Answer
    st.markdown(f"""
    <div class="assistant-message">
        <strong>🤖 Answer:</strong><br><br>
        {result['answer']}
    </div>
    """, unsafe_allow_html=True)
    
    # Citations
    if result.get('citations'):
        st.markdown("### 📚 Sources")
        for citation in result['citations']:
            similarity_pct = citation.get('similarity', 0) * 100
            st.markdown(f"""
            <div class="citation">
                <div class="citation-title">📄 {citation['document_title']}</div>
                <div style="color: #64748b; font-size: 0.85rem; margin-top: 0.25rem;">
                    Page {citation['page_number']} • Relevance: {similarity_pct:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)


# Main UI
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📚 Context-Aware RAG Agent</h1>
        <p>Advanced PDF Question Answering with Gemini AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Control Panel")
        
        # File upload
        st.markdown("#### 📤 Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF documents"
        )
        
        if st.button("🚀 Process PDFs", use_container_width=True):
            if uploaded_files:
                results = process_uploaded_files(uploaded_files)
                if results:
                    st.success(f"✅ Processed {len(results)} file(s)!")
                    for result in results:
                        if 'error' not in result:
                            st.info(f"📄 **{result['filename']}**\n- Pages: {result['pages']}\n- Chunks: {result['chunks']}")
            else:
                st.warning("⚠️ Please upload PDF files first")
        
        st.markdown("---")
        
        # Statistics
        if st.session_state.pipeline:
            st.markdown("#### 📊 Statistics")
            stats = st.session_state.pipeline.get_stats()
            
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{stats['total_chunks']}</div>
                <div class="stat-label">Total Chunks</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{len(st.session_state.processed_files)}</div>
                <div class="stat-label">Documents</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Clear database
        if st.button("🗑️ Clear Database", use_container_width=True):
            if st.session_state.pipeline:
                st.session_state.pipeline.clear_database()
                st.session_state.processed_files = []
                st.session_state.chat_history = []
                st.success("✅ Database cleared!")
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 Ask Questions")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="user-message">
                    <strong>👤 You:</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                display_answer(message['content'])
        
        # Question input
        question = st.text_input(
            "Ask a question about your documents:",
            placeholder="What is the main topic discussed in the document?",
            key="question_input"
        )
        
        col_ask, col_clear = st.columns([3, 1])
        
        with col_ask:
            if st.button("🔍 Ask Question", use_container_width=True):
                if question:
                    if not st.session_state.pipeline:
                        st.warning("⚠️ Please upload and process PDFs first")
                    else:
                        # Add question to history
                        st.session_state.chat_history.append({
                            'role': 'user',
                            'content': question
                        })
                        
                        # Get answer
                        with st.spinner("🤔 Thinking..."):
                            result = st.session_state.pipeline.ask_question(question)
                        
                        # Add answer to history
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': result
                        })
                        
                        # Rerun to display
                        st.rerun()
                else:
                    st.warning("⚠️ Please enter a question")
        
        with col_clear:
            if st.button("🔄 Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
    
    with col2:
        st.markdown("### ℹ️ How It Works")
        st.markdown("""
        1. **📤 Upload PDFs**: Upload your documents using the sidebar
        
        2. **🔄 Processing**: 
           - Text extraction with OCR
           - Smart chunking (600 words)
           - Embedding generation
           - Vector storage
        
        3. **❓ Ask Questions**: Type your questions in the chat
        
        4. **🎯 Get Answers**: 
           - Query expansion
           - Hybrid retrieval
           - Grounded answers with citations
        
        ---
        
        **✨ Features:**
        - 🔍 Semantic search
        - 📊 Hybrid retrieval
        - 🎯 Page citations
        - 🤖 Gemini AI powered
        """)


if __name__ == "__main__":
    main()
