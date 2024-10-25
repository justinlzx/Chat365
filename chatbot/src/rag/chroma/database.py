import traceback
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from typing import Any, Dict, List
import os
import shutil
import asyncio
import chromadb
from chromadb.config import Settings

DATA_PATHS = [
    "https://aws.amazon.com/blogs/opensource/deploy-large-language-models-easily-with-the-new-ezsmdeploy-python-sdk/"
]
PERSIST_DIRECTORY = "./embeddings"


def ensure_directory_permissions():
    """Ensure the persistence directory exists and has proper permissions"""
    if os.path.exists(PERSIST_DIRECTORY):
        # Remove existing directory to ensure clean state
        shutil.rmtree(PERSIST_DIRECTORY)

    # Create directory with explicit permissions
    os.makedirs(PERSIST_DIRECTORY, mode=0o777, exist_ok=True)

    # Ensure the directory and its parent have proper permissions
    try:
        os.chmod(PERSIST_DIRECTORY, 0o777)
        parent_dir = os.path.dirname(PERSIST_DIRECTORY)
        if parent_dir:
            os.chmod(parent_dir, 0o777)
    except Exception as e:
        print(f"Warning: Could not set permissions: {e}")


def initialize_embeddings() -> HuggingFaceEmbeddings:
    """Initialize HuggingFace embeddings with proper configuration"""
    return HuggingFaceEmbeddings()


def initialize_vector_store() -> Chroma:
    """Initialize Chroma vector store with the specified embedding function"""
    ensure_directory_permissions()

    # Initialize ChromaDB with explicit settings
    chroma_client = chromadb.PersistentClient(
        path=PERSIST_DIRECTORY,
        settings=Settings(
            allow_reset=True, is_persistent=True, anonymized_telemetry=False
        ),
    )

    # Initialize the vector store with the client
    return Chroma(
        client=chroma_client,
        collection_name="example_collection",
        embedding_function=initialize_embeddings(),
        persist_directory=PERSIST_DIRECTORY,
    )


def load_documents(paths: List[str]) -> List[Document]:
    """Synchronously load documents from paths"""
    documents = []
    for path in paths:
        loader = RecursiveUrlLoader(url=path)
        documents.extend(loader.load())
    return documents


def split_documents(docs: List[Any]) -> List[Any]:
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(docs)


def clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Clean metadata to ensure all values are of acceptable types"""
    cleaned = {}
    for key, value in metadata.items():
        if value is None:
            cleaned[key] = ""
        elif isinstance(value, (int, float, bool, str)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


def process_documents(vector_store: Chroma) -> Chroma:
    """Process documents and create embeddings"""
    try:
        # Load documents synchronously
        documents = load_documents(DATA_PATHS)  # Assuming this has a sync version

        # Process documents directly
        splits = split_documents(documents)
        cleaned_splits = filter_complex_metadata(
            splits, allowed_types=(str, int, float, bool)
        )

        # Add documents to vector store
        vector_store.add_documents(cleaned_splits)

        return vector_store

    except Exception as e:
        print(f"Error processing documents: {str(e)}")
        raise


def get_db() -> Chroma:
    """Get or create the vector database"""
    try:
        # Ensure persist directory exists
        if not os.path.exists(PERSIST_DIRECTORY):
            os.makedirs(PERSIST_DIRECTORY)
            print(f"Created persist directory at {PERSIST_DIRECTORY}")

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        # Create or load vector store
        vector_store = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
        )

        # Print collection stats for debugging
        collection_count = vector_store._collection.count()
        print(f"Collection count: {collection_count}")

        # If empty, process documents
        if collection_count == 0:
            print("Vector store is empty, processing documents...")
            process_documents(vector_store)
            print("Documents processed and added to vector store")

        return vector_store

    except Exception as e:
        print(f"Detailed error in get_db: {str(e)}")
        traceback.print_exc()  # Print the full traceback
        raise RuntimeError(f"Database error: {str(e)}")


if __name__ == "__main__":

    async def main():
        """Async main function to run the document processing"""
        try:
            db = initialize_vector_store()
            result = await process_documents(db)
            print("Documents processed successfully")
            print("Database initialized successfully")
            return result
        except Exception as e:
            print(f"Error processing documents: {str(e)}")
            return None

    asyncio.run(main())
