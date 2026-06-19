from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Ollama (LLM + embeddings) ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3.2:3b"
    ollama_condenser_model: str = "ministral-3:8b"
    ollama_embed_model: str = "nomic-embed-text"
    llm_temperature: float = 0.0

    # --- Chroma vector store ---
    chroma_dir: str = str(PROJECT_ROOT / 'data' / 'chroma')
    collection_name: str = "personal_rag"

    # --- Chunking (RecursiveCharacterTextSplitter) ---
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # --- Retrieval ---
    # candidates retrieved per leg (dense + BM25)
    top_k: int = 5
    # kept after cross-encoder rerank
    rerank_top_n: int = 3
    # EnsembleRetriever (RRF) weights: [semantic, BM25]
    hybrid_weights: list[float] = [0.6, 0.4]
    reranker_model: str = "BAAI/bge-reranker-base"

    # --- Local data directories ---
    uploads_dir: str = str(PROJECT_ROOT / 'data' / 'uploads')


# Import this singleton everywhere settings are needed.
settings = Settings()


if __name__ == '__main__':
    print(f'Ollama LLM: {settings.ollama_llm_model}')
    print(f'Chroma directory: {settings.chroma_dir}')