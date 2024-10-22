import asyncio
from langchain_chroma import Chroma

from .database import initialize_vector_store, process_documents, initialize_embeddings


def similarity_search(db, query: str) -> list:
    results = db.similarity_search(query, k=3)

    return results


if __name__ == "__main__":

    async def main():

        db = initialize_vector_store()
        process_documents(db)

        query = "What is the best way to do a rag model?"
        results = similarity_search(db, query)

        print(results)

    asyncio.run(main())