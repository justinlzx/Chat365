from langchain_chroma import Chroma
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from typing import Any, List, Dict
import os
import shutil

# DATA_PATHS = "https://www.healthhub.sg/programmes/nsc"  # add more sites as required
DATA_PATHS = "https://medium.com/@callumjmac/implementing-rag-in-langchain-with-chroma-a-step-by-step-guide-16fc21815339"


def initialize_embeddings() -> HuggingFaceEmbeddings:
    """
    Initialize HuggingFace embeddings with proper configuration
    """
    return HuggingFaceEmbeddings()


def initialize_vector_store(embedding_function: HuggingFaceEmbeddings) -> Chroma:
    """
    Initialize Chroma vector store with the specified embedding function
    """

    # Delete the directory if it exists
    if os.path.exists("./embeddings"):
        shutil.rmtree("./embeddings")

    # Ensure the directory is re-created
    os.makedirs("./embeddings", exist_ok=True)

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

    document_loader = RecursiveUrlLoader(data_path)
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
        documents = load_documents(DATA_PATHS)
        splits = split_documents(documents)

        cleaned_splits = filter_complex_metadata(splits, allowed_types=(str, int, float, bool))

        # Add documents to vector store
        vector_store.add_documents(cleaned_splits)

        # # Persist the vector store
        # vector_store.persist()

        return vector_store, documents

    except Exception as e:
        print(f"Error processing documents: {str(e)}")
        raise


def clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean metadata to ensure all values are of acceptable types
    """
    cleaned = {}
    for key, value in metadata.items():
        # Convert None to empty string
        if value is None:
            cleaned[key] = ""
        # Convert numbers to appropriate types
        elif isinstance(value, (int, float, bool, str)):
            cleaned[key] = value
        # Convert everything else to string
        else:
            cleaned[key] = str(value)
    return cleaned


if __name__ == "__main__":
    vector_store, documents = process_documents()

    # Print first document for inspection
    if documents:
        print("First document content:")
        print(documents[0].page_content)
        print("\nMetadata:")
        print(documents[0].metadata)
