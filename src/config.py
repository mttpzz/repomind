import os
import litellm
from dotenv import load_dotenv

load_dotenv()

# Langfuse configuration for automatic tracing via LiteLLM
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY", "")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY", "")
os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST", os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"))

# Enable LiteLLM callback to Langfuse (env vars alone are not read by litellm; must set the callback lists directly)
# "langfuse_otel" (not legacy "langfuse") is required for langfuse SDK v3+ installed here
litellm.success_callback = ["langfuse_otel"]
litellm.failure_callback = ["langfuse_otel"]

# Set local Ollama base URL for LiteLLM routing
os.environ["OLLAMA_API_BASE"] = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

SETTINGS = {
    "default_model": os.getenv("DEFAULT_MODEL", "anthropic/claude-haiku-4-5-20251001"),
    "local_model": os.getenv("LOCAL_MODEL", "ollama/gpt-oss:20b"),
    "embedding_model": os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
    "vector_db_path": "./data/chroma_db"
}