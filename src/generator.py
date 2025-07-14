"""
This file handles generating responses using a language model (LLM). It takes user questions and relevant documents, builds a prompt, and gets an answer from the LLM.
"""
try:
    import openai
except ImportError:
    openai = None
from typing import List, Dict

class ResponseGenerator:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo") -> None:
        # This function sets up the OpenAI client and model for generating responses.
        if openai is None:
            raise ImportError("openai package is not installed. Please install it to use ResponseGenerator.")
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def generate_response(self, query: str, context_documents: List[Dict]) -> str:
        """
        This function generates a response to the user's question using the provided context documents.
        It builds a prompt and sends it to the LLM, then returns the generated answer as a string.
        
        Args:
            query: This is the user's question as a string.
            context_documents: This is a list of relevant documents to help answer the question.
        Returns:
            The generated response as a string.
        """
        try:
            # This function builds a context string from the provided documents.
            context = self.build_context(context_documents)
            
            # This function creates a prompt for the LLM using the query and context.
            prompt = self.create_prompt(query, context)
            
            # This function sends the prompt to the OpenAI API and gets the response.
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided company information. Be accurate and cite sources when possible."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            # Defensive: handle possible missing or None fields
            if not response or not hasattr(response, 'choices') or not response.choices:
                return ""
            message = response.choices[0].message if hasattr(response.choices[0], 'message') else None
            content = getattr(message, 'content', None) if message else None
            return content or ""
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def build_context(self, documents: List[Dict]) -> str:
        """
        This function builds a single context string from a list of documents.
        Each document's content and source are included, separated by lines.
        """
        if not documents:
            return "No relevant information found."
        
        context_parts = []
        for i, doc in enumerate(documents):
            source = doc.get('metadata', {}).get('source', 'Unknown')
            content = doc.get('content', '')
            # This line adds the document's source and content to the context.
            context_parts.append(f"[Source: {source}]\n{content}\n")
        
        # This joins all document contexts together, separated by lines.
        context = "\n---\n".join(context_parts)
        
        # This limits the context length to avoid exceeding token limits for the LLM.
        max_context_length = 3000
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
            
        return context
    
    def create_prompt(self, query: str, context: str) -> str:
        """
        This function creates the final prompt for the LLM by combining the context and the user's question.
        It also includes instructions for the LLM to answer based only on the provided context.
        """
        prompt = f"""Based on the following company information, please answer the user's question accurately and comprehensively.

Context from company documents:
{context}

User Question: {query}

Instructions:
- Answer based only on the information provided in the context
- If the answer is not in the context, say so clearly
- Be specific and cite the source document when possible
- Keep the answer concise but complete

Answer:"""
        return prompt
