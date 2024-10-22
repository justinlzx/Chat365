from contextlib import asynccontextmanager
from ..rag.chroma.database import (
    get_db,
    initialize_embeddings,
    initialize_vector_store,
    process_documents,
)
from fastapi import FastAPI
from .routes import router
from dotenv import load_dotenv
from ..utils.logger import logger
import os

load_dotenv()


MODEL_PATH = os.getenv("MODEL_ID")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI application"""
    global _vector_store, _model, _tokenizer

    try:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # Initialize vector store
        logger.info("Initializing vector store...")
        _vector_store = get_db()

        # await process_documents(_vector_store)
        # Process documents if needed
        logger.info("Vector store initialized and documents successfully")

        yield
    finally:
        # # Cleanup
        # if _vector_store:
        #     _vector_store.persist()
        logger.info("Shutdown complete")


app = FastAPI(lifespan=lifespan)


app.include_router(router, prefix="/model", tags=["RagBot"])
