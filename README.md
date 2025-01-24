# 💬 Chat365

Chat365 is a real-time Retrieval Augmented Generation (RAG) chat application to support the digital transformation of Healthy365 — Healthiest365.

It is built using FastAPI, a modern, fast (high-performance), web framework for building APIs with Python, and integrates with OpenAI's GPT-4 for multilingual text generation capabilities.

RAG is implemented with Chroma as the vector store and LangChain for prompt chaining.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip (Python package installer)

### Installation

1. Clone the repository
    ```bash
    git clone https://github.com/justinlzx/Chat365.git
    cd Healthier365
    ```

2. Create a virtual environment (venv) to isolate the project dependencies:

   ```bash
   python -m venv env
   ```

3. Activate the virtual environment:

   - On Windows:
     ```bash
     env\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source env/bin/activate
     ```

4. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Set the necessary environment variables:

   Create a `.env` file in the project root directory and add the following variables:

   ```bash
   OPENAI_API_KEY=your-openai-api-key
   ```

6. Run the development server:

   ```bash
   python -B -m main --reload
   ```

   The `--reload` flag will automatically restart the server when you make changes to the code.

7. Open your web browser and navigate to `http://localhost:8000/gradio` to access the Chat Interface.
