from app.loaders import load_document
from app.vectorstore import vectorstore
from app.config import settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

splitter = RecursiveCharacterTextSplitter(
                                            chunk_size = settings.chunk_size,
                                            chunk_overlap = settings.chunk_overlap
)

def ingest_file(path: str) -> dict:

    docs = load_document(path)
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("Empty file uploaded. Please retry with another file.")
        return None
    
    for chunk_idx, chunk in enumerate(chunks):
        chunk.metadata['chunk_index'] = chunk_idx + 1

    vectorstore.add_documents(chunks)
    print('Chunks added to vector-store succesfully!')

    return {'filename': Path(path).name, 'chunk_count': chunk_idx + 1}

if __name__ == '__main__':
    path = './data/uploads/test.pdf'
    response = ingest_file(path)
    print(response)
    vectorstore.reset_collection()