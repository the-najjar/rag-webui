# RAG Interface Web Application

This project is a web-based RAG (Retrieval-Augmented Generation) interface that allows users to upload documents, ask questions about them, and receive answers based on the content of those documents. The application also provides references to where the answers were found within the uploaded documents.

## Features

- Upload and process various document types
- Ask questions about uploaded documents
- Receive answers with references to source locations
- Supports multiple file formats

## Supported File Types

- PDF (.pdf)
- CSV (.csv)
- Excel (.xlsx, .xls)
- Text (.txt)
- JSON (.json)

## Prerequisites

- [Ollama](https://ollama.ai/) installed on your system
- Python 3.x

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/rag-webui.git
   cd rag-webui
   ```

2. Install Ollama following the instructions on the [Ollama website](https://ollama.ai/).

3. Install an LLM model using Ollama:
   ```
   ollama pull <model_name>
   ```
   Example:
   ```
   ollama pull llama2:13b
   ```

4. Start the Ollama server:
   ```
   ollama serve
   ```

5. Configure the application:
   - Open the `.env` file in the project root
   - Set the `OLLAMA_MODEL_ID` to the model you installed:
     ```
     OLLAMA_MODEL_ID=llama2:13b
     ```

## Usage

1. Run the application:
   - On macOS or Linux:
     ```
     ./run_macos_linux.sh
     ```
   - On Windows:
     ```
     run_windows.bat
     ```

2. Open your web browser and navigate to the provided local URL (typically `http://localhost:8000`).

3. Upload your documents using the web interface.

4. Ask questions about the uploaded documents and receive answers with source references.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the [MIT License](LICENSE).
