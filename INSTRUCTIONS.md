# RAG System Take-Home Assignment

## Overview
Build a Retrieval-Augmented Generation (RAG) system that helps employees find information from company documents. This assignment should take 4-6 hours to complete.

**You are encouraged to use AI coding assistants** (ChatGPT, Claude, GitHub Copilot, etc.) to help with implementation. We're interested in your problem-solving approach and RAG understanding, not just coding from scratch.

## The Problem
Create a system that can answer questions about company information by retrieving relevant document chunks and generating comprehensive responses.

## Deliverables
1. **Working RAG system** with a simple interface
2. **Documentation** (README.md) explaining your approach and design decisions
3. **Demo preparation** - be ready to show it running locally via screen-share

## Requirements

### Core System (Must Have)
- **Document Processing**: Handle at least 3 different file types (PDF, TXT, CSV recommended)
- **Chunking Strategy**: Implement meaningful text chunking (explain your approach)
- **Vector Storage**: Use any vector database (Chroma, FAISS, Pinecone, etc.)
- **Retrieval**: Similarity search returning top-k relevant chunks
- **Generation**: Integrate with an LLM API (OpenAI, Anthropic, etc.) for response generation
- **Simple Interface**: Web UI (Streamlit/Gradio) or command-line interface

### Evaluation Component (Important)
- Create 5-10 test questions covering different types of queries
- Show how your system handles these queries
- Document what works well and what limitations you observed
- Document what tools/library you use and how you made your selection of the tools

### Documentation
Your README should include:
- Setup/installation instructions
- Architecture overview and design decisions
- Chunking strategy explanation
- Sample queries and expected behavior
- Known limitations and potential improvements

**NOTE: We are looking to learn about how you work and your approach to solving a problem The more details you can share, the more we can get insights about you!**  

## Bonus Features (Optional)
- Conversation context/follow-up questions
- Query rewriting or expansion
- Metadata filtering (by document type, date, etc.)
- Comparison of different embedding models or chunking approaches
- Source attribution in responses

## Technical Constraints
- Use any programming language (Python recommended)
- Any LLM API is fine (you may need to create free accounts)
- Any vector database or embedding model
- Keep API costs minimal (we're not evaluating on scale)

## Provided Materials
We'll provide:
- Sample document set (10-15 company documents)
- Suggested test queries
- Basic project structure template (optional to use)

## Submission
- Share your code repository (GitHub preferred)
- Include clear setup instructions
- Be prepared for a 15-20 minute demo session where you'll:
  - Show the system running
  - Walk through your code architecture
  - Demonstrate with sample queries
  - Discuss design decisions and trade-offs

## Evaluation Focus
We're looking for:
- **Functional system** that retrieves and generates reasonable responses
- **Clear thinking** about RAG design choices and trade-offs
- **Code quality** - readable, organized, with proper error handling
- **Problem-solving approach** - how you tackle challenges and iterate
- **Understanding** of RAG limitations and potential improvements

## Questions?
Feel free to reach out if you need clarification on requirements or technical setup issues.
