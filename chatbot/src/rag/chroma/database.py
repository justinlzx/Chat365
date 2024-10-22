from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from typing import Any, AsyncGenerator, Dict, List
from asyncio import to_thread
import os
import shutil
import asyncio
import chromadb
from chromadb.config import Settings

DATA_PATHS = "https://aws.amazon.com/blogs/opensource/deploy-large-language-models-easily-with-the-new-ezsmdeploy-python-sdk/"
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


async def load_documents(data_path: str) -> List[Any]:
    """Load HTML documents from the specified URL asynchronously"""
    document_loader = RecursiveUrlLoader(url=data_path)
    return await to_thread(document_loader.load)


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


async def process_documents(vector_store: Chroma) -> Chroma:
    """Process documents and create embeddings"""
    try:
        # Load documents asynchronously
        documents = await load_documents(DATA_PATHS)

        # Process documents in thread pool
        splits = await to_thread(split_documents, documents)
        cleaned_splits = await to_thread(
            filter_complex_metadata, splits, allowed_types=(str, int, float, bool)
        )

        # Add documents to vector store in thread pool to avoid blocking
        await to_thread(vector_store.add_documents, cleaned_splits)

        # Persist the changes
        vector_store.persist()

        return vector_store

    except Exception as e:
        print(f"Error processing documents: {str(e)}")
        raise


async def get_db() -> AsyncGenerator[Chroma, None]:
    """Database dependency injection"""
    try:
        vector_store = initialize_vector_store()
        processed_store = await process_documents(vector_store)
        yield processed_store
    except Exception as e:
        raise RuntimeError(f"Database error: {str(e)}")
    finally:
        # Ensure proper cleanup
        if "vector_store" in locals():
            vector_store.persist()


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
