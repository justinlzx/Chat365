from langchain_chroma import Chroma

from .database import process_documents, initialize_embeddings


def similarity_search(db: Chroma, query: str) -> list:
    results = db.similarity_search(query, k=3)

    return results


if __name__ == "__main__":
    db, _ = process_documents()

    embedding_function = initialize_embeddings()

    query = "What is the best way to do a rag model?"
    results = similarity_search(db, query, embedding_function)

    print(results)
