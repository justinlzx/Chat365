from time import sleep
from typing import Generator

from langchain_chroma import Chroma

from ..rag.chroma.retriever import similarity_search
from ..rag.chroma.database import get_db
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_model_response(query: str, db: Chroma = None) -> Generator[str, None, None]:
    try:
        # Get relevant documents

        if not db:
            db = get_db()

        docs = similarity_search(db, query=query)

        system_prompt = """
You are a helpful health assistant whose job is to assist the user with answering queries only relating to their health. If the user asks a question that is not related to health or how to use health applications, politely remind them that you can only answer health-related questions.

If a question does not make any sense or is not factually coherent, explain why instead of answering something incorrectly. If you don't know the answer to a question, don't share false information.

Answer the user's questions as if you were holding a conversation with them. 

Respond in the same language as the user.
"""

        document_context = "\n\n".join([doc.page_content for doc in docs])
        # document_context = ""
        user_prompt = f"""The user asked:

{query}

Consider the following documents to help answer the user's question:

{document_context}"""

        model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.5,
            api_key=OPENAI_API_KEY,
            streaming=True,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Use regular streaming instead of async
        for chunk in model.stream(messages):
            sleep(0.1)  # Use regular time.sleep instead of asyncio.sleep
            if chunk.content:
                yield chunk.content

    except Exception as e:
        raise Exception(f"Error occurred: {str(e)}")
