from create_database import initialize_embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np


def embed_query(query: str, embedding_function: HuggingFaceEmbeddings) -> np.ndarray:
    """
    Embed a query using the specified embedding function
    """
    embedding_function = initialize_embeddings()
    return embedding_function.embed_query(query)


if __name__ == "__main__":
    # Initialize embedding function

    # Embed query
    query = "What is the best way to lose weight?"
    query_embedding = embed_query(query, embed_query)
    print(query_embedding)
