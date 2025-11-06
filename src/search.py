import os
from dotenv import load_dotenv
from src.vector_store import VectorStore
from langchain_groq import ChatGroq
from src.embedding import EmbeddingPipeline
from src.data_loader import load_all_documents

load_dotenv()

class RAGSearch:
    def __init__(self, collection_name: str = "chroma_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "gemma2-9b-it"):
        
        self.vector_store = VectorStore(collection_name)
        
        # Load or build vectorstore
        persist_dir = self.vector_store.persist_directory
        chroma_path = os.path.join(persist_dir, "chroma")

        if not os.path.exists(chroma_path):
            # load documents and build vector store
            print(f"[INFO] Loaded documents from data directory.")
            
            documents = load_all_documents("data")
            
            # Create embeddings and add to vector store
            embedding_pipeline = EmbeddingPipeline()
            chunked_docs = embedding_pipeline.chunk_documents(documents)
            embeddings = embedding_pipeline.embed_chunks(chunked_docs)

            self.vector_store.add_documents(chunked_docs, embeddings)
        else:
            self.vector_store._initialize_store()

        ### Initialize Groq LLM
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(api_key=groq_api_key, model="openai/gpt-oss-20b", temperature=0.1, max_tokens=500)

        print(f"[INFO] Groq LLM initialized: {llm_model}")


    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vector_store.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke([prompt])
        return response.content
