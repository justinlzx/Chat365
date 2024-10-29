import os
from typing import Optional, Dict, Any, List
import openai

# from tenacity import retry, stop_after_attempt, wait_exponential


class OpenAI:
    """
    A wrapper class for the OpenAI API client with additional functionality
    and error handling.
    """

    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        streaming: bool = False,
        max_retries: int = 3,
        timeout: int = 30,
        organization: Optional[str] = None,
    ):
        """
        Initialize the OpenAI API wrapper.

        Args:
            model: The model to use for completions
            temperature: Controls randomness in the output
            api_key: OpenAI API key
            streaming: Whether to stream the response
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            organization: OpenAI organization ID
        """
        self.client = openai.OpenAI(
            api_key=api_key, organization=organization, timeout=timeout
        )
        self.model = model
        self.temperature = temperature
        self.streaming = streaming
        self.max_retries = max_retries

    # @retry(
    #     stop=stop_after_attempt(3),
    #     wait=wait_exponential(multiplier=1, min=4, max=10),
    #     retry_error_callback=lambda retry_state: retry_state.outcome.result(),
    # )
    # async def complete(
    #     self, messages: List[Dict[str, str]], **kwargs
    # ) -> Dict[str, Any]:
    #     """
    #     Send a completion request to the OpenAI API with retry logic.

    #     Args:
    #         messages: List of message dictionaries
    #         **kwargs: Additional parameters to pass to the API

    #     Returns:
    #         API response dictionary
    #     """
    #     try:
    #         response = await self.client.chat.completions.create(
    #             model=self.model,
    #             messages=messages,
    #             temperature=self.temperature,
    #             stream=self.streaming,
    #             **kwargs,
    #         )

    #         if self.streaming:
    #             return self._handle_streaming_response(response)
    #         return self._handle_normal_response(response)

    #     except openai.APIError as e:
    #         print(f"OpenAI API Error: {str(e)}")
    #         raise
    #     except Exception as e:
    #         print(f"Unexpected error: {str(e)}")
    #         raise

    def _handle_normal_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Process a normal (non-streaming) API response."""
        return {
            "content": response.choices[0].message.content,
            "role": response.choices[0].message.role,
            "finish_reason": response.choices[0].finish_reason,
            "model": response.model,
            "usage": response.usage._asdict() if response.usage else None,
        }

    def _handle_streaming_response(self, response: Any) -> Dict[str, Any]:
        """Process a streaming API response."""
        collected_content = []
        for chunk in response:
            if chunk.choices[0].delta.content:
                collected_content.append(chunk.choices[0].delta.content)
                if self.streaming:
                    print(chunk.choices[0].delta.content, end="", flush=True)

        return {
            "content": "".join(collected_content),
            "role": "assistant",
            "finish_reason": chunk.choices[0].finish_reason,
            "model": chunk.model,
        }

    async def get_token_count(self, text: str) -> int:
        """
        Get the token count for a given text using the tiktoken library.

        Args:
            text: The text to count tokens for

        Returns:
            Number of tokens in the text
        """
        try:
            import tiktoken

            encoding = tiktoken.encoding_for_model(self.model)
            return len(encoding.encode(text))
        except ImportError:
            print("tiktoken not installed. Install with: pip install tiktoken")
            return -1

    def update_config(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        streaming: Optional[bool] = None,
    ) -> None:
        """
        Update the configuration parameters of the wrapper.

        Args:
            model: New model to use
            temperature: New temperature value
            streaming: New streaming setting
        """
        if model is not None:
            self.model = model
        if temperature is not None:
            self.temperature = temperature
        if streaming is not None:
            self.streaming = streaming
