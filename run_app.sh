#!/bin/bash
# Set environment variable to suppress tokenizer warnings
export TOKENIZERS_PARALLELISM=false

# Run the Streamlit app
streamlit run main.py