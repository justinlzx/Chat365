import asyncio
from typing import List
from fastapi import FastAPI
import gradio as gr
from ..service import get_model_response


def query(message: str, history: List[List[str]] = None) -> str:
    """
    Streaming query function for Gradio interface
    """
    full_response = ""
    try:
        # Explicitly annotate as a generator function
        for chunk in get_model_response(message):
            if chunk:
                full_response += chunk
                yield full_response
    except Exception as e:
        raise gr.Error(f"An error occurred: {str(e)}")


def mount_gradio_interface(app: FastAPI, path: str = "/"):
    interface = gr.ChatInterface(
        fn=query,
        title="Health Companion",
        examples=[
            "What are common symptoms of the flu?",
            "How can I maintain a healthy diet?",
        ],
        retry_btn=None,
        undo_btn=None,
        clear_btn="Clear",
    )

    gradio = gr.mount_gradio_app(app, interface, path=path)

    return gradio
