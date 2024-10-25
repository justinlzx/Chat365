import asyncio
from langchain_chroma import Chroma

from .database import initialize_vector_store, process_documents


def similarity_search(db, query: str) -> list:
    results = db.similarity_search(query, k=5)

    print(results)

    return results


if __name__ == "__main__":

    async def main():

        db = initialize_vector_store()
        process_documents(db)

        query = "What is the best way to do a rag model?"
        results = similarity_search(db, query)

        print(
            [
                {
                    "item": i+1,
                    "source:": results[i].metadata.source,
                    "page:": results[i].metadata.page,
                }
            ] for i in range(len(results))
        )

    asyncio.run(main())
