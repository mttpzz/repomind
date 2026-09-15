import os
from pathlib import Path
import chromadb

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank

from src.config import SETTINGS

class AdvancedRAGPipeline:
    """Manages Advanced RAG pipeline using LlamaIndex, ChromaDB, local Ollama embeddings, and a local Cross-Encoder Reranker."""
    
    def __init__(self, target_dir: str):
        self.target_dir = Path(target_dir)
        
        # 1. Configure global LlamaIndex settings with local Ollama embeddings
        Settings.embed_model = OllamaEmbedding(
            model_name=SETTINGS["embedding_model"],
            base_url=os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
        )
        # Disable default LlamaIndex LLM for chunking/summarization unless explicitly needed
        Settings.llm = None 

        # 2. Initialize persistent ChromaDB client
        self.db_client = chromadb.PersistentClient(path=SETTINGS["vector_db_path"])
        self.chroma_collection = self.db_client.get_or_create_collection("repomind_codebase")
        
        # 3. Setup LlamaIndex vector store integration
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

    def ingest_codebase(self):
        """Loads and indexes the target codebase directory into ChromaDB via LlamaIndex."""
        if not self.target_dir.exists():
            raise FileNotFoundError(f"Target directory {self.target_dir} does not exist.")

        print(f"Loading files from {self.target_dir}...")
        
        # Load source code files recursively (excluding hidden folders)
        reader = SimpleDirectoryReader(
            input_dir=str(self.target_dir),
            required_exts=['.py', '.js', '.ts', '.md'],
            exclude_hidden=True
        )
        documents = reader.load_data()

        # Build index and persist nodes into ChromaDB
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context
        )
        print(f"Successfully indexed {len(documents)} document nodes into ChromaDB.")

    def get_query_engine(self, top_k: int = 10, top_n: int = 5):
        """
        Creates a query engine equipped with a Reranker.
        Retrieves top_k initial nodes from vector search, then reranks them down to top_n using a Cross-Encoder.
        """
        # Load existing index if already created
        if not hasattr(self, 'index'):
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                storage_context=self.storage_context
            )

        # Initialize local Cross-Encoder Reranker (e.g., ms-marco or bge-reranker)
        reranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-v2-m3",
            top_n=top_n
        )

        # Build query engine with similarity top_k and node postprocessors (reranker)
        query_engine = self.index.as_query_engine(
            similarity_top_k=top_k,
            node_postprocessors=[reranker]
        )
        
        return query_engine