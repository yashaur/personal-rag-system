from langchain_ollama import ChatOllama, OllamaEmbeddings
from app.config import settings

CHAT_MODEL = settings.ollama_llm_model
CONDENSER_MODEL = settings.ollama_condenser_model
BASE_URL = settings.ollama_base_url
EMBED_MODEL = settings.ollama_embed_model
TEMPERATURE = settings.llm_temperature

llm = ChatOllama(
                model = CHAT_MODEL,
                base_url = BASE_URL,
                temperature = TEMPERATURE
)

condenser_llm = ChatOllama(
                model = CONDENSER_MODEL,
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

    response = llm.invoke("Hi, how are you")

    print(response)