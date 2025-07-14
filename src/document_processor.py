"""
Document processing and chunking functionality
"""
import os
import pandas as pd
from typing import List, Dict
import PyPDF2

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_documents(self, documents_dir: str) -> List[Dict]:
        """
        Load all documents from the specified directory

        Returns:
            List of dictionaries with 'content', 'source', and 'metadata'
        """
        documents = []

        for filename in os.listdir(documents_dir):
            file_path = os.path.join(documents_dir, filename)
            
            if not os.path.isfile(file_path):
                continue
                
            file_extension = filename.lower().split('.')[-1]
            
            try:
                if file_extension == 'txt':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                elif file_extension == 'csv':
                    content = self.process_csv(file_path)
                    
                elif file_extension == 'pdf':
                    content = self.process_pdf(file_path)
                    
                else:
                    print(f"Skipping unsupported file type: {filename}")
                    continue
                    
                # Create document with metadata
                document = {
                    'content': content,
                    'source': filename,
                    'metadata': {
                        'file_type': file_extension,
                        'file_path': file_path,
                        'file_name': filename,
                        'created_date': os.path.getctime(file_path)  # Add file creation date
                    }
                }
                documents.append(document)
                print(f"Loaded: {filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                
        print(f"Total documents loaded: {len(documents)}")
        return documents

    def chunk_text(self, text: str, metadata: Dict = {}, strategy: str = "fixed") -> List[Dict]:
        """
        Split text into chunks using the specified strategy.
        - 'fixed': Fixed-size chunking with overlap (default)
        - 'retrieval_tree': Hierarchical chunking (parent/child)
        - 'agentic': Semantic/section-based chunking
        - 'sentence': Sentence-boundary chunking
        - 'structure': Document structure-aware chunking

        Args:
            text: Text to chunk
            metadata: Additional metadata to include with each chunk
            strategy: Chunking strategy ('fixed', 'retrieval_tree', 'agentic', 'sentence', or 'structure')


        Returns:
            List of text chunks with metadata
        """
        if strategy == "sentence":
            return self._sentence_boundary_chunking(text, metadata)
        if strategy == "structure":
            return self._structure_aware_chunking(text, metadata)
        if strategy == "retrieval_tree":
            return self._retrieval_tree_chunking(text, metadata)
        if strategy == "agentic":
            return self._agentic_chunking(text, metadata)
        
        # Default: fixed-size chunking with overlap
        chunks = []
        start = 0
        chunk_index = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_content = text[start:end]
            chunk_data = {
                'content': chunk_content,
                'chunk_index': chunk_index,
                'start_char': start,
                'end_char': min(end, len(text)),
            }
            if metadata:
                chunk_data['metadata'] = metadata
            chunks.append(chunk_data)
            start += self.chunk_size - self.chunk_overlap
            chunk_index += 1
            if start >= len(text):
                break
        return chunks

    def _sentence_boundary_chunking(self, text: str, metadata: Dict = {}) -> List[Dict]:
        """
        Sentence-boundary chunking: split text into chunks at sentence boundaries, keeping each chunk under a character limit.
        """
        import re
        max_chunk_size = self.chunk_size
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        chunk_index = 0
        start_char = 0
        for sentence in sentences:
            if not sentence.strip():
                continue
            if len(current_chunk) + len(sentence) + 1 > max_chunk_size and current_chunk:
                end_char = start_char + len(current_chunk)
                chunk_data = {
                    'content': current_chunk.strip(),
                    'chunk_index': chunk_index,
                    'start_char': start_char,
                    'end_char': end_char
                }
                if metadata:
                    chunk_data['metadata'] = metadata
                chunks.append(chunk_data)
                chunk_index += 1
                start_char += len(current_chunk) + 1
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        # Add last chunk
        if current_chunk.strip():
            end_char = start_char + len(current_chunk)
            chunk_data = {
                'content': current_chunk.strip(),
                'chunk_index': chunk_index,
                'start_char': start_char,
                'end_char': end_char
            }
            if metadata:
                chunk_data['metadata'] = metadata
            chunks.append(chunk_data)
        return chunks

    def _structure_aware_chunking(self, text: str, metadata: Dict = {}) -> List[Dict]:
        """
        Document structure-aware chunking: split based on sections, subsections, bullets, and numbered lists.
        """
        import re
        lines = text.splitlines()
        chunks = []
        current_chunk = []
        chunk_index = 0
        start_char = 0
        char_pointer = 0
        structure_pattern = re.compile(r"^(\s*[A-Z][A-Z0-9 \-:]{4,}|\s*\d+\.|\s*\d+\)|\s*\-|\s*\*)")
        for line in lines:
            if structure_pattern.match(line) and current_chunk:
                chunk_text = "\n".join(current_chunk).strip()
                if chunk_text:
                    end_char = char_pointer
                    chunk_data = {
                        'content': chunk_text,
                        'chunk_index': chunk_index,
                        'start_char': start_char,
                        'end_char': end_char,
                        'structure_header': current_chunk[0].strip()
                    }
                    if metadata:
                        chunk_data['metadata'] = metadata
                    chunks.append(chunk_data)
                    chunk_index += 1
                    start_char = char_pointer
                current_chunk = [line]
            else:
                current_chunk.append(line)
            char_pointer += len(line) + 1  # +1 for newline
        # Add last chunk
        if current_chunk:
            chunk_text = "\n".join(current_chunk).strip()
            if chunk_text:
                end_char = char_pointer
                chunk_data = {
                    'content': chunk_text,
                    'chunk_index': chunk_index,
                    'start_char': start_char,
                    'end_char': end_char,
                    'structure_header': current_chunk[0].strip()
                }
                if metadata:
                    chunk_data['metadata'] = metadata
                chunks.append(chunk_data)
        return chunks


    def _retrieval_tree_chunking(self, text: str, metadata: Dict = {}) -> List[Dict]:
        """
        Hierarchical retrieval tree chunking: split into large parent chunks, then split each parent into smaller child chunks.
        """
        parent_chunk_size = 2000
        parent_overlap = 400
        child_chunk_size = 1000
        child_overlap = 200
        parent_chunks = []
        start = 0
        parent_index = 0
        # First, split into parent chunks
        while start < len(text):
            end = start + parent_chunk_size
            parent_content = text[start:end]
            parent_chunks.append({
                'content': parent_content,
                'parent_index': parent_index,
                'start_char': start,
                'end_char': min(end, len(text)),
            })
            start += parent_chunk_size - parent_overlap
            parent_index += 1
            if start >= len(text):
                break
        # Now, split each parent chunk into child chunks
        all_child_chunks = []
        for parent in parent_chunks:
            p_content = parent['content']
            p_start = parent['start_char']
            child_start = 0
            child_index = 0
            while child_start < len(p_content):
                child_end = child_start + child_chunk_size
                child_content = p_content[child_start:child_end]
                chunk_data = {
                    'content': child_content,
                    'parent_index': parent['parent_index'],
                    'chunk_index': child_index,
                    'start_char': p_start + child_start,
                    'end_char': min(p_start + child_end, parent['end_char']),
                }
                if metadata:
                    chunk_data['metadata'] = metadata
                all_child_chunks.append(chunk_data)
                child_start += child_chunk_size - child_overlap
                child_index += 1
                if child_start >= len(p_content):
                    break
        return all_child_chunks

    def _agentic_chunking(self, text: str, metadata: Dict = {}) -> List[Dict]:
        """
        Agentic chunking: split text on semantic section boundaries (e.g., headings, Q/A, steps).
        """
        import re
        lines = text.splitlines()
        chunks = []
        current_chunk = []
        current_title = None
        chunk_index = 0
        start_char = 0
        char_pointer = 0
        # Regex for section boundaries: ALL CAPS, numbered steps, Q:/A:
        section_pattern = re.compile(r"^(\s*[A-Z][A-Z0-9 \-:]{4,}|\s*Q:|\s*A:|\s*Step \d+|^\d+\.|^\d+\))")
        for line in lines:
            if section_pattern.match(line) and current_chunk:
                chunk_text = "\n".join(current_chunk).strip()
                if chunk_text:
                    end_char = char_pointer
                    chunk_data = {
                        'content': chunk_text,
                        'chunk_index': chunk_index,
                        'start_char': start_char,
                        'end_char': end_char,
                        'section_title': current_title or "Section"
                    }
                    if metadata:
                        chunk_data['metadata'] = metadata
                    chunks.append(chunk_data)
                    chunk_index += 1
                    start_char = char_pointer
                current_chunk = [line]
                current_title = line.strip()
            else:
                current_chunk.append(line)
            char_pointer += len(line) + 1  # +1 for newline
        # Add last chunk
        if current_chunk:
            chunk_text = "\n".join(current_chunk).strip()
            if chunk_text:
                end_char = char_pointer
                chunk_data = {
                    'content': chunk_text,
                    'chunk_index': chunk_index,
                    'start_char': start_char,
                    'end_char': end_char,
                    'section_title': current_title or "Section"
                }
                if metadata:
                    chunk_data['metadata'] = metadata
                chunks.append(chunk_data)
        return chunks

    def process_csv(self, file_path: str) -> str:
        """Convert CSV to text representation"""
        df = pd.read_csv(file_path)
    
        text_lines = []
        
        text_lines.append(f"CSV Data from {os.path.basename(file_path)}:")
        text_lines.append("-" * 50)
        
        text_lines.append(f"Columns: {', '.join(df.columns)}")
        text_lines.append(f"Total rows: {len(df)}")
        text_lines.append("")
        
        for idx, row in df.iterrows():
            row_text = []
            for col, value in row.items():
                if pd.notna(value):
                    row_text.append(f"{col}: {value}")
            text_lines.append(" | ".join(row_text))
            text_lines.append("")  
            
        return "\n".join(text_lines)


    def process_pdf(self, file_path: str) -> str:
        """Extract text from PDF"""

        text_content = []

        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)

            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()

                if text.strip():
                    text_content.append(f"---- Page {page_num + 1} ----")
                    text_content.append(text)
        
        return "\n".join(text_content)

