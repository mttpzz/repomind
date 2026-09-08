from litellm import completion, embedding
from src.config import SETTINGS

class LLMFactory:
    """Factory Pattern for centralized creation and management of LLM and Embedding calls via LiteLLM (supporting Anthropic and local Ollama models)."""
    
    @staticmethod
    def call_llm(prompt: str, system_prompt: str = "You are an expert AI Engineer specialized in codebases.", model: str = None):
        target_model = model or SETTINGS["default_model"]
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # LiteLLM handles routing to Anthropic or local Ollama and automatically traces to Langfuse
        response = completion(model=target_model, messages=messages)
        return response.choices[0].message.content

    @staticmethod
    def get_embedding(text: str):
        # Generate embeddings locally using nomic-embed-text via Ollama/LiteLLM
        response = embedding(model=SETTINGS["embedding_model"], input=[text])
        return response["data"][0]["embedding"]