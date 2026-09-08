# RepoMind 🧠🤖

**RepoMind** is an enterprise-grade, advanced Multi-Agent RAG (Retrieval-Augmented Generation) assistant designed to ingest, analyze, and query complex codebases. Built with state-of-the-art Generative AI frameworks, it combines high-performance vector search, cross-encoder reranking, autonomous multi-agent collaboration, and robust MLOps observability.

---

## 🚀 Key Features & Tech Stack

- **Advanced RAG & Orchestration:** Powered by **LlamaIndex** for intelligent document loading and data structuring.
- **Cross-Encoder Reranking:** Integrates **`BAAI/bge-reranker-v2-m3`** (`top_n = 5`) to boost semantic retrieval precision and filter out irrelevant noise.
- **Multi-Agent Collaboration:** Uses **CrewAI** to orchestrate specialized agents (Senior Codebase Analyst & Technical Documentation Specialist) working sequentially.
- **Vector Database:** Uses **ChromaDB** with persistent disk storage for efficient similarity search.
- **Unified LLM Gateway & MLOps:** 
  - **LiteLLM** handles model routing and provides a unified interface.
  - **Langfuse** provides automatic end-to-end tracing and monitoring for every LLM call and agent workflow step.
- **Flexible Model Support:** Powered by **Anthropic Claude Haiku (`anthropic/claude-haiku-4-5-20251001`)** for cloud tasks, alongside local open-weight models via **Ollama** (`gpt-oss:20b` and `nomic-embed-text` for embeddings).
- **Software Design Patterns:** Implements the **Factory Pattern** (`LLMFactory`) for clean separation of concerns and maintainable code architecture.
- **Containerization:** Fully dockerized for seamless deployment.

---

## 📁 Project Architecture

repomind/
├── .env.example                # Environment variables template
├── Dockerfile                  # Containerization configuration
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
├── README.md                   # Project documentation
├── src/
│   ├── __init__.py
│   ├── config.py               # Centralized configuration & Langfuse setup
│   ├── core/
│   │   ├── __init__.py
│   │   └── llm_factory.py      # Factory Pattern for Anthropic & Ollama via LiteLLM
│   ├── rag/
│   │   ├── __init__.py
│   │   └── pipeline.py         # LlamaIndex + ChromaDB + BGE Reranker
│   ├── agents/
│   │   ├── __init__.py
│   │   └── crew_manager.py     # CrewAI Multi-Agent Orchestration
│   └── main.py                 # CLI Entry Point
└── tests/
    └── test_rag.py             # Pytest unit tests for pipeline initialization

---

## 🛠️ Getting Started & Installation

Follow these steps to set up and run **RepoMind** locally.

### 1. Prerequisites
- **Python 3.10+** installed on your machine.
- **Ollama** installed locally to serve open-source models and embeddings.

### 2. Clone and Install Dependencies
Clone the repository and install the required Python packages:

git clone https://github.com/your-username/repomind.git
cd repomind

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install --no-cache-dir -r requirements.txt

### 3. Configure Environment Variables
Copy the example environment file and fill in your API keys:

cp .env.example .env

Open `.env` and set your configuration:
- `ANTHROPIC_API_KEY`: Your Anthropic API key (`anthropic/claude-haiku-4-5-20251001`).
- `LANGFUSE_PUBLIC_KEY` & `LANGFUSE_SECRET_KEY`: Your Langfuse tracking keys (optional, but recommended for MLOps).
- Local model names and Ollama endpoint URLs.

### 4. Start Ollama and Pull Local Models
Make sure your Ollama background service is running:

ollama serve

In a separate terminal window, download the required local models (embedding and local LLM):

ollama pull nomic-embed-text
ollama pull gpt-oss:20b

---

## 🧪 Running Tests

RepoMind includes unit tests to ensure architectural stability. Run the test suite using `pytest`:

pytest

---

## 💻 Usage (CLI Guide)

RepoMind comes with a built-in Command Line Interface (CLI) supporting two primary actions: **Ingestion** and **Querying**.

### Step A: Ingest a Codebase
Scan, chunk, and index a target source code directory into ChromaDB:

python -m src.main ingest --dir ./path/to/your/target/codebase

### Step B: Query the Codebase with Multi-Agent Analysis
Run the multi-agent workflow to analyze the codebase and answer technical prompts:

python -m src.main query --target-dir ./path/to/your/target/codebase --prompt "How is authentication handled in this project?"

---

## 📊 MLOps & Observability

RepoMind integrates **LiteLLM** and **Langfuse** natively. Every query, vector retrieval, reranking step, and multi-agent interaction is automatically logged and tracked. You can inspect token usage, latency, and step-by-step reasoning chains directly on your Langfuse Dashboard (https://cloud.langfuse.com).

---

## 🐳 Docker Deployment

To build and run the application inside a Docker container:

docker build -t repomind:latest .
docker run --env-file .env -v $(pwd)/data:/app/data repomind:latest

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.