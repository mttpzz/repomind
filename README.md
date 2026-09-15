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
- **Containerization:** Fully dockerized for seamless deployment.

---

## 🛠️ Getting Started & Installation

Follow these steps to set up and run **RepoMind** locally.

### 1. Prerequisites
- **Python 3.12+** installed on your machine (pinned dependency versions are verified against 3.12).
- **Ollama** installed locally to serve open-source models and embeddings.

### 2. Clone and Install Dependencies
Clone the repository and install the required Python packages:

git clone https://github.com/mttpzz/repomind.git
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

In a separate terminal window, download the required local embedding model:

ollama pull nomic-embed-text

---

## 🧪 Running Tests

RepoMind includes unit tests covering the RAG pipeline setup (`tests/test_rag.py`), the CrewAI agent/LLM wiring (`tests/test_crew_manager.py`), and the CLI entry point (`tests/test_main.py`). All external calls (CrewAI, LiteLLM, Ollama embeddings) are mocked, so the suite runs offline — no Anthropic key, Langfuse account, or running Ollama instance required. Run it with `pytest`:

pytest

Note: these tests only verify that the code wires the right calls with the right parameters — they don't catch integration issues (wrong model name, incompatible library version, misconfigured API key). Always do a real end-to-end `ingest` + `query` run (see below) before trusting a change.

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

Build the image:

docker build -t repomind:latest .

The container's entrypoint runs `python -m src.main`, so the `ingest`/`query` subcommand and flags are passed at `docker run` time. Ollama runs on your **host**, not inside the container, so point `OLLAMA_API_BASE` at the host (`host.docker.internal` on Docker Desktop for Windows/Mac). The target codebase you want to analyze also isn't part of the image — mount it as a volume.

### Ingest

docker run --rm --env-file .env \
  -v "$(pwd)/data:/app/data" \
  -v "/path/to/target/codebase:/target" \
  -e OLLAMA_API_BASE=http://host.docker.internal:11434 \
  repomind:latest ingest --dir /target

### Query

docker run --rm --env-file .env \
  -v "$(pwd)/data:/app/data" \
  -v "/path/to/target/codebase:/target" \
  -e OLLAMA_API_BASE=http://host.docker.internal:11434 \
  repomind:latest query --target-dir /target --prompt "How is authentication handled in this project?"

- `-v $(pwd)/data:/app/data` persists the ChromaDB vector store (`./data/chroma_db`) across `ingest` and `query` runs — without it, each container starts with an empty index.
- `-v /path/to/target/codebase:/target` mounts the codebase you want to ingest/query; `--dir`/`--target-dir` must point at the path *inside* the container (`/target`), not the host path.
- `--env-file .env` supplies `ANTHROPIC_API_KEY` and the optional Langfuse keys — don't bake secrets into the image.

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.
