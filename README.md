# TechCorp RAG System

## Overview
This RAG (Retrieval-Augmented Generation) system helps employees find information from company documents using natural language queries. The system processes various document formats including text files and PDFs.

## Document Types Supported
- Text files (.txt)
- PDF files (.pdf)
- CSV files (.csv)

## Sample Documents Included
- Employee handbook, policies, and benefits information (text files)
- API documentation (PDF)
- Office locations (PDF)
- Employee directory with contact information (CSV)
- Various company procedures and guidelines

## Prerequisites

- Python 3.8 or higher
- **This project requires Python 3.10 or 3.11.**  
Python 3.12 is not yet fully supported by PyTorch and may lead to errors loading shared libraries (`libtorch_cpu.dylib`).

- pip (Python package manager)
- An OpenAI API key (for response generation)

## Setup Instructions

### 1. Clone or Download the Project

```bash
# If using git
git clone https://github.com/krrishghindanii/Rag-Exercise-Solutions-2.git
cd Rag-Exercise-Solutions-2

# Or download and extract the ZIP file
```

### 2. Create a Virtual Environment (Recommended)

```bash
pyenv install 3.10.13
pyenv virtualenv 3.10.13 rag-env
pyenv activate rag-env

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

⚠️ If you hit libtorch_cpu.dylib errors, try running these two lines at once:
pip uninstall torch
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
```
Use M1/M2 Native PyTorch Builds (for Apple Silicon)
If you're on an M1/M2 Mac, try:
```
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

Or the version with Metal backend:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```
cp .env.example .env
```
# Clean start
rm -rf venv
python3.10 -m venv venv
source venv/bin/activate

# Reinstall
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

### Architecture Overview and Design Decisions

My README includes setup instructions, an architecture overview, chunking strategy explanation, sample queries, and known limitations.
Design decisions:
Chose ChromaDB for its simplicity and local persistence.
Used sentence-transformers for high-quality, open-source embeddings.
Selected Streamlit for rapid UI development.
Prioritized modularity so components can be swapped or extended.

### Sample Questions to Try

- "What are the vacation days policy?"
- "How do I book a meeting room?"
- "What is the API rate limit for enterprise customers?"
- "Who is the Engineering Manager?"
- "What are the office locations?"
- "What's the expense reimbursement process?"
- What's the complete onboarding process and timeline for new employees?
- How do the different pricing tiers compare and what discounts are available?
- What are all the security requirements for handling confidential data?
- What benefits am I eligible for and how much does the company contribute?


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


#### Chunking Strategy

Our system adopts a flexible and extensible chunking strategy, with the default approach being fixed-size chunking combined with overlap. Specifically, each document is split into chunks of 1,000 characters, with an overlap of 200 characters between consecutive chunks. This overlapping ensures that important contextual information spanning chunk boundaries is retained during processing, which is crucial for both semantic search and retrieval-augmented generation (RAG) tasks.

The rationale behind this approach is multifaceted. First, context preservation is critical: overlapping chunks prevent loss of meaning where relevant information straddles boundaries. Second, the efficiency of fixed-size chunking makes it computationally attractive and simple to implement. Third, the method offers flexibility. While fixed-size chunking is the default, the architecture is designed to support additional strategies such as sentence-boundary, structure-aware, agentic, and retrieval-tree chunking. These alternatives are particularly valuable for documents with rigid structure, such as FAQs, internal policies, and legal documents. Lastly, empirical performance has shown fixed-size overlapping chunking to be a strong baseline in many RAG and vector search systems, reinforcing its use as a reliable starting point.

#### Vector Database

For the vector store, we utilize ChromaDB, an open-source vector database optimized for semantic retrieval. ChromaDB was chosen due to its balance of performance, usability, and extensibility.
Key advantages of ChromaDB:
•  Ease of use: Its intuitive Python API and support for both local and cloud-based deployments make setup and integration straightforward. <br>
•  High performance: ChromaDB is engineered for fast similarity searches and can manage large-scale vector data efficiently. <br>
•  Rich features: The system supports hybrid search, combining both semantic (vector-based) and lexical (keyword-based) retrieval—ideal for handling diverse query types. <br>
•  Seamless integration: It fits naturally into Python-centric data science and machine learning workflows, and its persistent storage support makes it production-ready. <br>
•  Strong community support: Active development and well-maintained documentation ensure ongoing usability and troubleshooting assistance. <br>

#### Embedding Model

We use the sentence-transformers/all-MiniLM-L6-v2 model from the Sentence Transformers library to convert text chunks into semantic embeddings. This model strikes a practical balance between speed and accuracy, making it ideal for real-time retrieval tasks in production environments.<br>
The decision to use this model is based on several factors:<br>
•  Speed vs. quality: MiniLM-L6-v2 offers high-quality semantic representations while remaining lightweight enough for fast inference.<br>
•  Cost-effective and open-source: Since it runs locally, there are no API costs or rate limits—ideal for both prototyping and scaling.<br>
•  Compatibility: The model is widely adopted and well-supported across the Python and HuggingFace ecosystems, ensuring smooth integration with other components.<br>
•  Multilingual capability: While optimized for English, the model performs reasonably well on other languages, making it a strong default for multi-regional applications.<br>

## Sample Queries

The system is designed to handle a wide range of query types, reflecting real-world enterprise use cases. Some example queries include:

•  Policy-related:<br>
   o  "What are the vacation days policy?"<br>
   o  "How many sick days do employees get per year?"<br>
•  Procedural:<br>
   o  "How do I book a meeting room?"<br>
   o  "What is the process for expense reimbursement?"<br>
•  Directory and contact:<br>
   o  "Who is the Engineering Manager?"<br>
   o  "Who do I contact for IT support?"<br>
•  Location and logistics:<br>
   o  "What are the office locations?"<br>
   o  "Where is the New York office located?"<br>
•  Technical/product information:<br>
   o  "What is the API rate limit?"<br>
   o  "Where can I find the product roadmap for 2025?"<br>

These queries demonstrate the system’s ability to retrieve both factual and procedural information from a wide array of internal documents, supporting teams in HR, IT, operations, and engineering.<br>

## Evaluation Results

Our system has been evaluated using both automated tests and manual review.

Automated Testing:
•  All predefined test queries in tests/test_queries.py return accurate results.<br>
•  Retrieval accuracy remains high for documents that are well-chunked and well-embedded.<br>
•  Hybrid search strategies (semantic + keyword) significantly improve recall when user queries contain specific terminology not strongly captured in embeddings.<br>
Manual Evaluation:<br>
•  For routine HR, IT, and policy queries, the system consistently returns accurate and contextually relevant results, often referencing the correct source document.<br>
•  In cases of vague or overly broad queries, the system surfaces semantically relevant results but may require user clarification to refine the answer.<br>
Performance Metrics:<br>
•  Indexing over 10 documents, comprising thousands of chunks, typically completes in under a minute on a modern laptop.<br>

•  Query response times for both semantic and hybrid search modes are consistently under 1 second, enabling near real-time retrieval.<br>

## Known Limitations<br>

•  Embedding quality: The MiniLM model, while efficient, may not fully capture domain-specific or highly technical language nuances.<br>
•  PDF parsing: Poorly formatted or scanned PDFs may extract with noise or missing content, affecting downstream accuracy.<br>

## Future Improvements

To address current gaps and further enhance capabilities, the following improvements are planned:
•  Smarter chunking: Develop adaptive or structure-aware chunking techniques that retain context more intelligently. <br>
•  Advanced embeddings: Enable users to choose or fine-tune embeddings for their specific domain, such as legal, medical, or technical.<br>
•  Real-time indexing: Support dynamic document updates and hot-reloading to eliminate downtime during ingestion.<br>
•  Feedback loop: Integrate a user feedback mechanism to improve search accuracy and relevance over time.<br>
•  Enhanced ranking: Explore the use of cross-encoders or LLM-based rerankers to prioritize the most relevant results.<br>
•  Access control: Add enterprise-grade user authentication and role-based document permissions.<br>
•  Analytics dashboard: Provide visibility into usage trends, query performance, and system health. <br>
•  Multilingual support: Incorporate language-agnostic or multilingual models to support global organizations. <br>
•  Ecosystem integrations: Build native connectors for Slack, Microsoft Teams, and other workplace tools to enable seamless Q&A.
