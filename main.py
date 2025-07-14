"""
Main Streamlit application for RAG system
"""
import streamlit as st
import os
from dotenv import load_dotenv
from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingManager
from src.retriever import DocumentRetriever
from src.generator import ResponseGenerator
from src.config import Config

# Load environment variables
load_dotenv()

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.dp = None
    st.session_state.em = None
    st.session_state.dr = None
    st.session_state.rg = None

def initialize_system():
    """Initialize all components of the RAG system"""
    with st.spinner("Initializing RAG system..."):
        # Initialize components
        st.session_state.dp = DocumentProcessor()
        st.session_state.em = EmbeddingManager()
        st.session_state.dr = DocumentRetriever()
        
        # Check if OpenAI API key is available
        if not Config.OPENAI_API_KEY:
            st.error("OpenAI API key not found! Please set OPENAI_API_KEY in your .env file.")
            return False
        
        st.session_state.rg = ResponseGenerator(Config.OPENAI_API_KEY)
        
        # Check if ChromaDB needs to be populated
        if st.session_state.dr.collection.count() == 0:
            st.info("First time setup: Loading and indexing documents...")
            
            # Load documents
            docs = st.session_state.dp.load_documents(Config.DOCUMENTS_DIR)
            
            # Process all documents
            all_chunks = []
            progress_bar = st.progress(0)
            
            for i, doc in enumerate(docs):
                chunks = st.session_state.dp.chunk_text(
                    doc['content'],
                    {'source': doc['source'], **doc['metadata']}
                )
                all_chunks.extend(chunks)
                progress_bar.progress((i + 1) / len(docs))
            
            # Generate embeddings
            st.write(f"Generating embeddings for {len(all_chunks)} chunks...")
            texts = [chunk['content'] for chunk in all_chunks]
            embeddings = st.session_state.em.generate_embeddings(texts)
            
            # Store in ChromaDB
            st.session_state.dr.add_documents(all_chunks, embeddings.tolist())
            st.success(f"Successfully indexed {len(all_chunks)} document chunks!")
        else:
            st.success(f"RAG system ready! ({st.session_state.dr.collection.count()} chunks indexed)")
        
        st.session_state.initialized = True
        return True

def search_and_generate(query, file_type=None, date_from=None):
    """Perform search and generate response with filters (no to-date)"""
    with st.spinner("Searching for relevant information..."):
        # Generate query embedding
        query_embedding = st.session_state.em.embed_query(query)
        # Search for relevant documents with filters
        results = st.session_state.dr.search(query_embedding.tolist(), top_k=Config.TOP_K_DOCUMENTS, file_type=file_type, date_from=date_from)
        # If no results, try hybrid search (semantic + keyword)
        if not results:
            results = st.session_state.dr.hybrid_search(query, query_embedding.tolist(), top_k=Config.TOP_K_DOCUMENTS)
        # Display retrieved documents
        with st.expander("📚 Retrieved Documents", expanded=False):
            for i, result in enumerate(results):
                source = result['metadata'].get('source', 'Unknown')
                similarity = result.get('similarity_score', 0)
                st.write(f"**{i+1}. {source}** (similarity: {similarity:.3f})")
                st.text(result['content'][:200] + "...")
                st.divider()
    try:
        with st.spinner("Generating response..."):
            # Generate response
            response = st.session_state.rg.generate_response(query, results)
    except Exception as e:
        if "invalid_api_key" in str(e) or "Incorrect API key" in str(e):
            response = "Error: Invalid OpenAI API key. Please check your .env file and restart the app."
        else:
            response = f"Error generating response: {str(e)}"
    return response, results

def main():
    st.set_page_config(
        page_title="TechCorp Knowledge Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 TechCorp Knowledge Assistant")
    st.write("Ask questions about company policies, procedures, and information.")
    
    # Initialize system if not already done
    if not st.session_state.initialized:
        if not initialize_system():
            st.stop()
    
    # Sidebar for filters
    with st.sidebar:
        st.header("Filter Documents")
        file_type = st.selectbox("File type", options=["All", "pdf", "csv", "txt"], index=0)
        date_from = st.date_input("From date (file created)", value=None, key="date_from_sidebar")

    # Main columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Query interface
        query = st.text_input("Enter your question:", placeholder="e.g., What are the vacation days policy?")
        
        col_search, col_clear = st.columns([1, 5])
        with col_search:
            search_button = st.button("🔍 Search", type="primary")
        with col_clear:
            if st.button("🗑️ Clear"):
                st.session_state.clear()
                st.rerun()
    
    with col2:
        # Sample queries
        st.subheader("Sample Questions")
        sample_queries = [
            "What are the vacation days policy?",
            "How do I book a meeting room?",
            "What is the API rate limit?",
            "Who is the Engineering Manager?",
            "What are the office locations?",
            "What's the expense reimbursement process?"
        ]
        
        for sample in sample_queries:
            if st.button(sample, key=f"sample_{sample}"):
                query = sample
                search_button = True
    
    # Process query
    if search_button and query:
        # Convert file_type and dates for backend
        file_type_val = None if file_type == "All" else file_type
        date_from_val = None
        import datetime
        if date_from:
            date_from_val = datetime.datetime.combine(date_from, datetime.time.min).timestamp()
        response, results = search_and_generate(query, file_type_val, date_from_val)
        
        # Display response
        st.divider()
        st.subheader("💡 Answer")
        st.write(response)
        
        # Add feedback section
        st.divider()
        col_feedback1, col_feedback2 = st.columns(2)
        with col_feedback1:
            if st.button("👍 Helpful"):
                st.success("Thank you for your feedback!")
        with col_feedback2:
            if st.button("👎 Not Helpful"):
                st.info("Thank you for your feedback. We'll work on improving our responses.")

if __name__ == "__main__":
    main()
