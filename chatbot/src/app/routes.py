import sys
from fastapi import APIRouter
from ..rag.chroma.database import process_documents
from ..rag.chroma.retriever import similarity_search
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
import os

router = APIRouter()


@router.post("/query")
def query(db, query: str):
    docs = similarity_search(db, query=query)

    system_prompt = SystemMessage(
        content="""
        You are a helpful health assistant whose job is to assist the user with answering queries relating to their health. Here are some relevant documents about healthcare that you can use to help the user.
        
        If a question does not make any sense or is not factually coherent, 
        explain why instead of answering something incorrectly. If you don't 
        know the answer to a question, don't share false information.
        \n\n
        """,
    )

    user_prompt = HumanMessage(
        content="""The user asked:
        \n\n
        {input}
        \n\n
        Consider the following documents to help answer the user's question:
        \n\n
        """
    )

    rag_prompt = "".join([doc.page_content for doc in docs])

    final_prompt = system_prompt + user_prompt + rag_prompt

    final_prompt.format_messages(input=query)

    print(final_prompt)


if __name__ == "__main__":
    db, documents = process_documents()
    user_query = "What is the best way to do a rag model?"
    result = query(db, user_query)
    print(result)
