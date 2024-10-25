import asyncio
from time import sleep, time
from ..rag.chroma.retriever import similarity_search
from fastapi import HTTPException
from ..rag.chroma.database import process_documents, get_db
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")



def get_query(query: str, db):
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
            streaming=True
        )

        # Create messages for chat completion
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        
        for chunk in chain.stream(messages):
            # print(chunk)
            time.sleep(10)
            # sleep(10)
            yield chunk
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

