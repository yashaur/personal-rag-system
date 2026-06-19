from app.loaders import load_document
from app.vectorstore import vectorstore, get_ids_by_filename, retrieve_doc_list
from app.config import settings
from app.retrieval import refresh
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

splitter = RecursiveCharacterTextSplitter(
                                            chunk_size = settings.chunk_size,
                                            chunk_overlap = settings.chunk_overlap
)

uploads_dir = settings.uploads_dir


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


def delete_single_file(filename: str) -> dict:    
    count_ids, ids = get_ids_by_filename(filename)

    if not ids:
        return {'deleted': False, 'message': f"'{filename}': No such file exists.", 'chunks_deleted': 0}
    
    vectorstore.delete(ids)
    refresh()

    (Path(uploads_dir) / Path(filename)).unlink(missing_ok = True)
    return {'deleted': True, 'message': f"Succesfully deleted '{filename}'",  'chunks_deleted': count_ids}


def delete_all_files() -> dict:
    doc_list = retrieve_doc_list()
    doc_names = list(doc_list.keys())
    N_chunks = sum([doc_list[doc]['chunks'] for doc in doc_list])

    vectorstore.reset_collection()
    refresh()

    for doc in doc_names:
        (Path(uploads_dir) / Path(doc)).unlink(missing_ok = True)

    return {'deleted': True, 'message': 'All files deleted.', 'chunks_deleted': N_chunks}


if __name__ == '__main__':
    path = './data/uploads/test.pdf'
    # response = ingest_file(path)
    # print(response)
    # vectorstore.reset_collection()