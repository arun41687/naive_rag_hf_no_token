"""Document ingestion and indexing module for RAG system."""

import os
import json
from typing import List, Dict, Tuple
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class DocumentIngestor:
    """Handles PDF parsing and text chunking."""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Initialize the document ingestor.
        
        Args:
            chunk_size: Number of characters per chunk
            overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def parse_pdf(self, pdf_path: str, doc_name: str) -> List[Dict]:
        """
        Parse a PDF file into chunks with metadata.
        
        Args:
            pdf_path: Path to the PDF file
            doc_name: Name of the document (e.g., "Apple 10-K")
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = []
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                full_text = ""
                page_mapping = {}  # Map character position to page number
                
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text() or ""
                    start_pos = len(full_text)
                    full_text += page_text + "\n\n"
                    end_pos = len(full_text)
                    page_mapping[(start_pos, end_pos)] = page_num
            
                # Split into overlapping chunks
                for i in range(0, len(full_text), self.chunk_size - self.overlap):
                    chunk_text = full_text[i:i + self.chunk_size]
                    
                    if len(chunk_text.strip()) < 50:  # Skip very small chunks
                        continue
                    
                    # Find which page(s) this chunk spans
                    page_num = 1
                    for (start, end), page in page_mapping.items():
                        if start <= i < end:
                            page_num = page
                            break
                    
                    chunks.append({
                        "id": f"{doc_name}_{len(chunks)}",
                        "text": chunk_text,
                        "document": doc_name,
                        "page": page_num,
                        "position": i
                    })
            
            return chunks
        except Exception as e:
            raise RuntimeError(f"Error parsing PDF {pdf_path}: {str(e)}") from e


class VectorStore:
    """Manages embeddings and vector search."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector store.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
    
    def add_chunks(self, chunks: List[Dict]) -> None:
        """
        Add chunks to the vector store.
        
        Args:
            chunks: List of chunk dictionaries
        """
        self.chunks.extend(chunks)
        
        # Generate embeddings
        texts = [chunk["text"] for chunk in self.chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # Create FAISS index
        embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.index.add(embeddings.astype(np.float32))
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (chunk, distance) tuples
        """
        if self.index is None or not self.chunks:
            raise RuntimeError("Vector store is empty. Add chunks before searching.")
        
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            results.append((self.chunks[idx], float(distance)))
        
        return results
    
    def save(self, save_dir: str) -> None:
        """Save the vector store."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(save_dir, "index.faiss"))
        
        # Save chunks metadata
        with open(os.path.join(save_dir, "chunks.json"), "w") as f:
            json.dump(self.chunks, f)
    
    def load(self, save_dir: str) -> None:
        """Load the vector store."""
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(save_dir, "index.faiss"))
        
        # Load chunks
        with open(os.path.join(save_dir, "chunks.json"), "r") as f:
            self.chunks = json.load(f)
