from transformers import AutoModelForCausalLM
from dotenv import load_dotenv
import os

load_dotenv()


def init_model(model_path: str) -> AutoModelForCausalLM:
    """
    Initialize the model from the specified path
    """
    HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

    return AutoModelForCausalLM.from_pretrained(model_path, token=HUGGING_FACE_TOKEN)


if __name__ == "__main__":
    model_path = "meta-llama/Meta-Llama-Guard-2-8B"
    model = init_model(model_path)
    print(model)
