import os
import json
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq

class RAGEngine:
    def __init__(self, groq_api_key: str):
        """Initialize RAG Engine with embedding model and Groq client"""
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.embedding_dim = 768  # Dimension of all-mpnet-base-v2
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.chunks = []  # Store text chunks with metadata
        
        # Initialize Groq client
        self.groq_client = Groq(api_key=groq_api_key)
        
        # Paths for persistence
        self.index_path = "vector_store/faiss.index"
        self.chunks_path = "vector_store/chunks.json"
        
        # Create vector_store directory
        os.makedirs("vector_store", exist_ok=True)
        
        # Load existing data if available
        self.load_index()
        
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
                
        return chunks
    
    def add_documents(self, documents: List[Dict[str, str]]):
        """
        Add documents to the vector store
        documents: List of dicts with 'text' and 'filename' keys
        """
        for doc in documents:
            text = doc['text']
            filename = doc['filename']
            
            # Split into chunks
            text_chunks = self.chunk_text(text)
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(text_chunks, show_progress_bar=False)
            
            # Add to FAISS index
            self.index.add(np.array(embeddings).astype('float32'))
            
            # Store chunks with metadata
            for i, chunk in enumerate(text_chunks):
                self.chunks.append({
                    'text': chunk,
                    'filename': filename,
                    'chunk_id': len(self.chunks)
                })
        
        # Save index and chunks
        self.save_index()
        print(f"Added {len(documents)} documents. Total chunks: {len(self.chunks)}")
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5, similarity_threshold: float = 1.5) -> List[Dict]:
        """
        Retrieve most relevant chunks for a query
        Returns empty list if no relevant chunks found
        """
        if len(self.chunks) == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search in FAISS
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)
        
        # Filter by similarity threshold (lower distance = more similar)
        relevant_chunks = []
        for dist, idx in zip(distances[0], indices[0]):
            if dist < similarity_threshold and idx < len(self.chunks):
                chunk_data = self.chunks[idx].copy()
                chunk_data['similarity_score'] = float(dist)
                relevant_chunks.append(chunk_data)
        
        return relevant_chunks
    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate answer using Groq LLM based on retrieved context"""
        if not context_chunks:
            return "I don't have enough information in the uploaded documents to answer this question."
        
        # Build context from chunks
        context = "\n\n".join([
            f"[From {chunk['filename']}]\n{chunk['text']}" 
            for chunk in context_chunks
        ])
        
        # Create prompt for Groq
        prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context. 
If the answer cannot be found in the context, you must respond with: "I don't have enough information in the uploaded documents."

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            # Call Groq API
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions strictly based on provided context. Never make up information."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama-3.3-70b-versatile",  # Fast and reliable model
                temperature=0.1,  # Low temperature for factual responses
                max_tokens=500
            )
            
            answer = chat_completion.choices[0].message.content
            return answer
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def query(self, question: str) -> Dict:
        """
        Main query method - retrieves context and generates answer
        """
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(question, top_k=5)
        
        # Generate answer
        answer = self.generate_answer(question, relevant_chunks)
        
        # Prepare response
        sources = []
        for chunk in relevant_chunks:
            sources.append({
                'filename': chunk['filename'],
                'text_snippet': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                'similarity_score': round(chunk['similarity_score'], 3)
            })
        
        return {
            'answer': answer,
            'sources': sources,
            'num_sources': len(sources)
        }
    
    def save_index(self):
        """Save FAISS index and chunks to disk"""
        faiss.write_index(self.index, self.index_path)
        with open(self.chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
    
    def load_index(self):
        """Load FAISS index and chunks from disk"""
        if os.path.exists(self.index_path) and os.path.exists(self.chunks_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.chunks_path, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            print(f"Loaded existing index with {len(self.chunks)} chunks")
    
    def clear_index(self):
        """Clear all documents from the vector store"""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.chunks = []
        self.save_index()
        print("Index cleared")
    
    def get_stats(self) -> Dict:
        """Get statistics about the current index"""
        return {
            'total_chunks': len(self.chunks),
            'total_documents': len(set(chunk['filename'] for chunk in self.chunks)) if self.chunks else 0
        }