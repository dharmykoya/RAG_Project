import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
import os
from src.data_loader import load_all_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter


class EmbeddingPipeline:
    """Handles document embedding geenration using SentenceTransformer."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', chunk_size: int = 1000, chunk_overlap: int = 200):
        """
            Ininitalize the EmbeddingManager with a specified SentenceTransformer model.

            Args:
                model_name (str): The name of the SentenceTransformer model to use.
                chunk_size (int): The size of each text chunk for embedding.
                chunk_overlap (int): The overlap size between text chunks. 
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
       
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk documents into smaller pieces for embedding.

        Args:
            documents (List[Dict[str, Any]]): List of documents to chunk.

        Returns:
            List[Dict[str, Any]]: List of chunked documents.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = splitter.split_documents(documents)
        print(f"Chunked {len(documents)} documents into {len(chunks)} chunks.")
        return chunks
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generate embeddings for a list of documents.

        Args:
            documents (List[Dict[str, Any]]): List of documents to embed.   
        Returns:
            List[Dict[str, Any]]: List of documents with embeddings.
        """
        texts = [chunk.page_content for chunk in chunks]
        print(f"Generating embeddings for {len(texts)} chunks using model {self.model_name}...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")

        return embeddings
