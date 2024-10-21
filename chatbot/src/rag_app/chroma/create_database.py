from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
import os


def initialize_embeddings() -> HuggingFaceEmbeddings:
    """
    Initialize HuggingFace embeddings with proper configuration
    """
    return HuggingFaceEmbeddings()


def initialize_vector_store(embedding_function: HuggingFaceEmbeddings) -> Chroma:
    """
    Initialize Chroma vector store with the specified embedding function
    """
    return Chroma(
        collection_name="example_collection",
        embedding_function=embedding_function,
        persist_directory="./embeddings",
    )


def load_documents(data_path):
    """
    Load HTML documents from the specified directory
    """
    # if not os.path.exists(DATA_PATH):
    #     raise FileNotFoundError(f"Directory not found: {DATA_PATH}")

    document_loader = WebBaseLoader(data_path)
    return document_loader.load()


def split_documents(docs):
    """
    Split documents into chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(docs)


def process_documents():
    """
    Main function to process documents and create embeddings
    """
    try:
        # Initialize embedding function
        embedding_function = initialize_embeddings()

        # Initialize vector store
        vector_store = initialize_vector_store(embedding_function)

        # Load and process documents
        documents = load_documents()
        splits = split_documents(documents)

        # Add documents to vector store
        vector_store.add_documents(splits)

        # # Persist the vector store
        # vector_store.persist()

        return vector_store, documents

    except Exception as e:
        print(f"Error processing documents: {str(e)}")
        raise


if __name__ == "__main__":
    vector_store, documents = process_documents()

    # Print first document for inspection
    if documents:
        print("First document content:")
        print(documents[0].page_content)
        print("\nMetadata:")
        print(documents[0].metadata)
