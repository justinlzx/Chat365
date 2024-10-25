from fastapi import APIRouter, HTTPException
from fastapi import Depends
from fastapi.responses import StreamingResponse

from .service import get_query
from ..rag.chroma.database import get_db

router = APIRouter()


@router.get("/query")
async def get_model_response(query: str, db=Depends(get_db)):
    """
    Endpoint for streaming chat responses
    """
    try:
        return StreamingResponse(
            get_query(query, db),
            media_type='text/event-stream'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


