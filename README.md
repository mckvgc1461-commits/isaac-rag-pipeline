# 🚀 ISAAC: Advanced Local RAG Pipeline

ISAAC is a high-performance, **100% local** Retrieval-Augmented Generation (RAG) system designed for secure scientific document analysis. It utilizes state-of-the-art LLMs and embedding models to provide grounded, accurate insights from technical PDFs without any data leakage.

## 🌟 Key Features
* **Deep PDF Parsing:** Powered by `PyMuPDF` for high-fidelity text and metadata extraction.
* **Privacy-First:** All processing is done locally via **Ollama (Llama 3)**. No external APIs used.
* **Scientific Accuracy:** Integrated with `BAAI/bge-small-en-v1.5` embeddings for precise semantic search.
* **Hallucination Guard:** Implements a confidence scoring system to ensure responses are strictly based on provided documents.
* **Smart Indexing:** Persistent storage for fast retrieval across large document sets.

## 🛠️ Tech Stack
* **Core Framework:** LlamaIndex
* **Inference Engine:** Ollama (Llama 3 8B)
* **Embeddings:** HuggingFace BGE-Small
* **Parser:** PyMuPDF (Fitz)

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have [Ollama](https://ollama.com/) installed and the Llama3 model pulled:
```bash
ollama pull llama3
2. Installation
Install the required Python dependencies:

Bash
pip install llama-index llama-index-llms-ollama llama-index-embeddings-huggingface pymupdf
3. Usage
Place your PDF documents in the ./documents folder.

Run the pipeline:

Bash
python main.py
📊 System Architecture
The system follows a 4-step pipeline:

Ingestion: Extracts text using specialized PDF readers.

Indexing: Converts text into 384-dimensional vectors.

Retrieval: Finds the most relevant document chunks based on query similarity.

Synthesis: Llama 3 generates an answer grounded only in the retrieved context.
