# Liber AI

## Chat with Your Documents

Welcome to Liber AI, a powerful tool that allows you to interact with your PDF documents using natural language queries. This application leverages advanced AI models to understand your questions and retrieve relevant information from your uploaded documents.

### Features

- **PDF Text Extraction**: Extract text from your uploaded PDF files.
- **Text Chunking**: Break down the extracted text into manageable chunks for efficient processing.
- **Embedding Creation**: Generate embeddings for text chunks using Google's Generative AI.
- **Similarity Search**: Retrieve relevant chunks based on user queries.
- **Question Answering**: Provide answers to user queries using Google Generative AI.

### Requirements

- **Ensure you have the following dependencies installed**:
python-dotenv
google-generativeai
PyPDF2
streamlit
langchain
langchain-community
langchain-google-genai
faiss-cpu

### Configuration

1. Create a `.env` file in the project directory.
2. Add your Google API key to the `.env` file:
    ```env
    GOOGLE_API_KEY=your_google_api_key_here
    ```

### Usage

1. Run the Streamlit application:
    ```sh
    streamlit run <your_script_name>.py
    ```
2. Open your web browser and navigate to the provided URL (usually `http://localhost:8501`).

### How It Works

1. **PDF Upload**: Upload a PDF file using the file uploader.
2. **Text Extraction**: The `get_pdf_text` function extracts text from the uploaded PDF.
3. **Text Chunking**: The `get_chunks` function splits the extracted text into chunks of 700 characters with an overlap of 100 characters.
4. **Embedding Creation**: The `chunk_embeddings` function generates embeddings for these chunks using Google's Generative AI model and saves them using FAISS.
5. **Query Processing**: When a user inputs a query, the `info_retrival` function retrieves relevant text chunks based on similarity search.
6. **Answer Generation**: The `gemini_response` function generates a response using the Google Generative AI model.

### Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    ```
2. Navigate to the project directory:
    ```sh
    cd liber-ai
    ```
3. Create a virtual environment:
    ```sh
    python -m venv venv
    ```
4. Activate the virtual environment:
    - On Windows:
      ```sh
      venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```sh
      source venv/bin/activate
      ```
5. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

