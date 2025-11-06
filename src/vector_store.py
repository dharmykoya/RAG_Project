import os
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
import pickle
import chromadb
import uuid



class VectorStore:
    """Manages a ChromaDB vector store for document embeddings."""

    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "../data/vector_store", embedding_model: str = "all-MiniLM-L6-v2"):
        """
            Initialize the VectorStore with a specified persistence directory.

            Args:
                collection_name (str): Name of the ChromaDB collection.
                persist_directory (str): Directory to persist the ChromaDB database.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.embedding_model = embedding_model
        self.model= SentenceTransformer(embedding_model)
        self._initialize_store()

    def _initialize_store(self):
        """Initialize the ChromaDB client and collection."""
        try:
            # Create the persistence directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            print(f"Initializing ChromaDB at {self.persist_directory}...")
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            # Get or create the collection
            self.collection = self.client.get_or_create_collection(name=self.collection_name, metadata={"description": "PDF Document Embeddings for RAG"})
            
            print(f"ChromaDB initialized successfully. Collection: {self.collection_name}")
            print(f"Number of existing documents in collection: {self.collection.count()}")
        
        except Exception as e:
            print(f"Error initializing Vector store: {e}")
            raise

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        """
            Add documents and their embeddings to the vector store.

            Args:
                documents (List[Dict[str, Any]]): List of document metadata dictionaries.
                embeddings (np.ndarray): Corresponding embeddings for the documents.
        """
        if not self.collection:
            raise ValueError("Vector store collection is not initialized.")
        
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match.")


        print(f"Adding {len(documents)} documents to the vector store...")

        # prepare data for ChromaDB
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate a unique ID for each document
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            # prepare metadata
            metadata = dict(doc.metadata)  # copy existing metadata
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            
            # Document Content
            documents_text.append(doc.page_content)

            #Embedding
            embeddings_list.append(embedding.tolist())

        # Add to ChromaDB collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Successfully added {len(documents)} documents to the vector store.")
            print(f"Total documents in collection after addition: {self.collection.count()}")

        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
            Query the vector store for similar documents.

            Args:
                query_embedding (np.ndarray): The embedding of the query.
                top_k (int): Number of top similar documents to retrieve.   
            Returns:
                List[Dict[str, Any]]: List of retrieved documents with metadata.
        """
        if not self.collection:
            raise ValueError("Vector store collection is not initialized.")

        try:
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k
            )

            retrieved_docs = []
            for i in range(len(results['ids'])):
                for j in range(len(results['ids'][i])):
                    doc_info = {
                        'id': results['ids'][i][j],
                        'document': results['documents'][i][j],
                        'metadata': results['metadatas'][i][j],
                        'distance': results['distances'][i][j]
                    }
                    retrieved_docs.append(doc_info)

            return retrieved_docs

        except Exception as e:
            print(f"Error querying vector store: {e}")
            raise

    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
            Search the vector store using a text query.

            Args:
                query (str): The text query.
                top_k (int): Number of top similar documents to retrieve.

            Returns:
                List[Dict[str, Any]]: List of retrieved documents with metadata.
        """
        query_embedding = self.model.encode([query])
        return self.search(query_embedding, top_k=top_k)         
        
