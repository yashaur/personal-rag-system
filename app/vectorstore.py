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

def get_ids_by_filename(filename: str) -> tuple[int, list[str]]:

    ids = vectorstore.get(where = {'source': filename})['ids']
    count_ids = len(ids)

    return count_ids, ids

def retrieve_doc_list() -> dict:

    vs = vectorstore.get()

    doc_list = {}

    for chunk in vs['metadatas']:
        source, chunk_index, page_number = chunk['source'], chunk['chunk_index'], chunk.get('page_number', 1)
        
        if source not in doc_list:
            doc_list[source] = doc_list.get(source, {'chunks': 1, 'pages': 1})

        doc_list[source]['chunks'] = max(doc_list[source]['chunks'], chunk_index)
        doc_list[source]['pages'] = max(doc_list[source]['pages'], page_number)

    return doc_list


if __name__ == '__main__':
    print(vectorstore.get()['documents'])
    print(len(vectorstore.get()['documents']))