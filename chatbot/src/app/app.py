from contextlib import asynccontextmanager

from src.app.gradio.app import mount_gradio_interface
from ..rag.chroma.database import (
    get_db,
    initialize_vector_store,
)
from fastapi import FastAPI
from .routes import router
from dotenv import load_dotenv
from ..utils.logger import logger
import os

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI application"""
    global _vector_store

    try:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Initialize vector store
        logger.info("Initializing vector store...")
        _vector_store = initialize_vector_store()
        # _vector_store = get_db()

        num_items = _vector_store._collection.count()
        # Process documents if needed
        logger.info(
            f"Vector store initialized and documents successfully. {num_items} in store "
        )

        yield
    finally:
        logger.info("Shutdown complete")


app = FastAPI(lifespan=lifespan)
app.include_router(router, prefix="/model", tags=["RagBot"])

app = mount_gradio_interface(app, "/gradio")
