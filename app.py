from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.vector_store import VectorStore
from src.search import RAGSearch


## Example usage
if __name__ == "__main__":
    # data_directory = "data"  # specify your data directory here
    # documents = load_all_documents(data_directory)
    # print(f"Loaded {len(documents)} documents from {data_directory}")   

    # embedding_pipeline = EmbeddingPipeline()
    # chunked_docs = embedding_pipeline.chunk_documents(documents)
    # embeddings = embedding_pipeline.embed_chunks(chunked_docs)

    # store = VectorStore("chroma_store")
    # store.add_documents(chunked_docs, embeddings)


    # print(f"Generated embeddings for {len(chunked_docs)} chunks.")
    # print(f"Embeddings shape: {embeddings}")

    rag_search = RAGSearch(collection_name="chroma_store", embedding_model="all-MiniLM-L6-v2", llm_model="gemma2-9b-it")
    query = "Who is Damilola Adekoya CV?"
    summary = rag_search.search_and_summarize(query, top_k=5)
    print(f"Summary for query '{query}':\n{summary}")

    # print("Sample document content:", documents)
