from app.config import settings
from app.llm import embeddings
from langchain_chroma import Chroma

COLLECTION_NAME = settings.collection_name
CHROMA_DIR = settings.chroma_dir

vectorstore = Chroma(
                    collection_name = COLLECTION_NAME,
                    persist_directory = CHROMA_DIR,
                    embedding_function = embeddings
)

if __name__ == '__main__':
    print(vectorstore.get()['documents'])
    print(len(vectorstore.get()['documents']))