from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI  # Changed from OpenAI to ChatOpenAI
from dotenv import load_dotenv
import os
from typing import List
from ..rag.chroma.database import process_documents, get_db
from ..rag.chroma.retriever import similarity_search

router = APIRouter()

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# class QueryResponse:
#     def __init__(self, response: str, sources: List[str]):
#         self.response = response
#         self.sources = sources


@router.post("/query")
async def query(query: str, db=Depends(get_db)):
    try:
        # Get relevant documents
        docs = similarity_search(db, query=query)

        # Create system and user prompts
        system_prompt = """You are a helpful health assistant whose job is to assist the user with answering queries relating to their health. Here are some relevant documents about healthcare that you can use to help the user.

If a question does not make any sense or is not factually coherent, explain why instead of answering something incorrectly. If you don't know the answer to a question, don't share false information."""

        # Format documents for context
        document_context = "\n\n".join([doc.page_content for doc in docs])

        user_prompt = f"""The user asked:

{query}

Consider the following documents to help answer the user's question:

{document_context}"""

        # Initialize ChatOpenAI
        model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.5,
            api_key=OPENAI_API_KEY,
        )

        # Create messages for chat completion
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        print(f"Messages: {messages}")

        # Generate response
        result = model.invoke(messages)

        # Extract sources for attribution
        sources = [doc.metadata.get("source", "Unknown source") for doc in docs]

        return JSONResponse(
            status_code=200,
            content={
                "message": "Response generated successfully",
                "response": result.content,
                "sources": sources,
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    db, documents = process_documents()
    user_query = "What is the best way to do a RAG model?"
    result = query(user_query, db)
    print(result)
