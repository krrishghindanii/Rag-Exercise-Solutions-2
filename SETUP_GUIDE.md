# TechCorp RAG System - Setup and Run Guide

This guide provides step-by-step instructions to set up and run the TechCorp RAG (Retrieval-Augmented Generation) system from scratch.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- An OpenAI API key (for response generation)

## Setup Instructions

### 1. Clone or Download the Project

```bash
# If using git
git clone https://github.com/krrishghindanii/Rag-Exercise-Solutions-2/blob/main/README.md
cd rag-interview-exercise-2-main

# Or download and extract the ZIP file
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `streamlit` - Web UI framework
- `openai` - OpenAI API client
- `chromadb` - Vector database
- `sentence-transformers` - Embedding generation
- `PyPDF2` - PDF processing
- `pandas` - CSV handling
- `python-dotenv` - Environment variable management
- `tiktoken` - Token counting

### 4. Set Up OpenAI API Key

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env file and add your OpenAI API key
# Open .env in a text editor and replace:
# OPENAI_API_KEY=your_openai_api_key_here
# with your actual OpenAI API key
```

### 5. Run the Application

#### Option A: Using Streamlit directly
```bash
streamlit run main.py
```

#### Option B: Using the run script (suppresses warnings)
```bash
# Make the script executable (macOS/Linux)
chmod +x run_app.sh

# Run the script
./run_app.sh
```

## First-Time Setup

When you run the application for the first time:

1. The system will automatically detect that the vector database is empty
2. It will load all documents from the `documents/` directory
3. Documents will be chunked into smaller pieces (1000 chars with 200 char overlap)
4. Embeddings will be generated for each chunk using sentence-transformers
5. All chunks will be stored in ChromaDB for fast retrieval

This process takes about 1-2 minutes and only happens once. The data is persisted in the `chroma_db/` directory.

## Using the Application

1. **Ask Questions**: Type your question in the text input field
2. **Use Sample Questions**: Click on any sample question button to try it
3. **Filter Results**: Use the dropdown to filter by file type (txt, pdf, csv) and select date ranges for file creation date. Only relevant results matching your filters will be shown.
4. **View Results**: 
   - The system will search for relevant document chunks
   - Generate a response using OpenAI's GPT-3.5-turbo
   - Display the answer with source attribution
5. **Check Sources**: Click "📚 Retrieved Documents" to see which documents were used

### New Feature: Metadata Filtering

- You can now filter search results by file type (txt, pdf, csv) and by file creation date.
- Use the dropdown and date pickers in the sidebar to narrow your search to specific document types or date ranges.
- This helps you find the most relevant information quickly, especially in large document sets.

### Sample Questions to Try

- "What are the vacation days policy?"
- "How do I book a meeting room?"
- "What is the API rate limit for enterprise customers?"
- "Who is the Engineering Manager?"
- "What are the office locations?"
- "What's the expense reimbursement process?"

## Troubleshooting

### Common Issues

1. **"No module named 'package_name'"**
   - Make sure you've activated your virtual environment
   - Run `pip install -r requirements.txt` again

2. **"OPENAI_API_KEY not found"**
   - Ensure you've created the `.env` file from `.env.example`
   - Check that your API key is correctly set in the `.env` file

3. **Tokenizer warnings**
   - These are harmless. Use `./run_app.sh` to suppress them
   - Or set: `export TOKENIZERS_PARALLELISM=false`

4. **Port already in use**
   - Kill existing Streamlit process: `pkill streamlit`
   - Or use a different port: `streamlit run main.py --server.port 8503`

### Resetting the System

To start fresh with a clean database:
```bash
# Remove the vector database
rm -rf chroma_db/

# Run the application again - it will rebuild the database
streamlit run main.py
```

## Project Structure

```
rag-interview-exercise-2-main/
├── documents/              # Company documents (TXT, PDF, CSV)
├── src/                    # Source code
│   ├── config.py          # Configuration settings
│   ├── document_processor.py  # Document loading and chunking
│   ├── embeddings.py      # Embedding generation
│   ├── retriever.py       # Vector search
│   └── generator.py       # LLM response generation
├── main.py                # Streamlit web application
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create from .env.example)
└── chroma_db/            # Vector database storage (created automatically)
```

## Performance Notes

- First query may be slower due to model loading
- Subsequent queries are faster as models are cached
- The system uses ~115 document chunks from 13 source files
- Embedding model: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- LLM: OpenAI GPT-3.5-turbo

## Support

If you encounter issues not covered here, please check:
1. Python version: `python3 --version` (should be 3.8+)
2. All dependencies installed: `pip list`
3. OpenAI API key is valid and has credits
4. Documents exist in the `documents/` directory
