from langchain_ollama import ChatOllama, OllamaEmbeddings
from app.config import settings

MODEL = settings.ollama_llm_model
BASE_URL = settings.ollama_base_url
EMBED_MODEL = settings.ollama_embed_model
TEMPERATURE = settings.llm_temperature

llm = ChatOllama(
                model = MODEL,
                base_url = BASE_URL,
                temperature = TEMPERATURE
)

embeddings = OllamaEmbeddings(
                                model = EMBED_MODEL,
                                base_url = BASE_URL
)

if __name__ == '__main__':
    vec = embeddings.embed_query("Are you running online or offline?")
    print(vec)
    print(f'vector length: {len(vec)}')

    print(llm.invoke("Hi, how are you").content)