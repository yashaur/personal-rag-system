from app.config import settings
from app.schemas import (
                            QueryRequest,
                            QueryResponse,
                            SourceDocument,
                            IngestResponse,
                            DocumentInfo,
                            DeleteResponse
                        )
from app.ingestion_deletion import ingest_file, delete_single_file, delete_all_files
from app.chains import answer_question, stream_answer_question
from app.vectorstore import retrieve_doc_list
import app.retrieval as retrieval

from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from fastapi.responses import StreamingResponse
from pathlib import Path
import json

router = APIRouter()

ALLOWED_FILETYPES = {'.pdf', '.txt'}

@router.post('/ingest', response_model = IngestResponse)
def ingest(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code = 400, detail = "Filename doesn't exist. Try again.")
    
    name = Path(file.filename).name
    ext = Path(name).suffix.lower()

    if ext not in ALLOWED_FILETYPES:
        raise HTTPException(status_code = 400, detail = f"Uploaded unsupported file type '{ext}'. Allowed: {ALLOWED_FILETYPES}")
    
    dest = Path(settings.uploads_dir) / name
    dest.parent.mkdir(parents = True, exist_ok = True)
    dest.write_bytes(file.file.read())

    try:
        result = ingest_file(str(dest))
    except ValueError as e:
        raise HTTPException(status_code = 400, detail = str(e))
    
    retrieval.refresh()

    return IngestResponse(**result)


@router.post('/query', response_model = QueryResponse)
def query(query_request: QueryRequest = Body(...)):
    question = query_request.question
    mode = query_request.mode
    session_id = query_request.session_id

    if mode == 'multi':
        chat_history = [m.model_dump() for m in query_request.chat_history]
    else:
        chat_history = None

    response = answer_question(question = question, chat_history = chat_history, session_id = session_id)

    sources = [SourceDocument(
                            content = doc.page_content,
                            source = doc.metadata['source'],
                            page_number = doc.metadata.get('page_number'))
                for doc in response['sources']
            ]
    
    return QueryResponse(answer = response['answer'], sources = sources)


@router.get('/documents', response_model = list[DocumentInfo])
def retrieve_docs():
    doc_list = retrieve_doc_list()
    return [DocumentInfo(filename = fname, page_count = doc_list[fname]['pages'], chunk_count = doc_list[fname]['chunks']) for fname in doc_list]


@router.post('/query/stream')
def query_stream(query_request: QueryRequest = Body(...)):
    question = query_request.question
    mode = query_request.mode
    session_id = query_request.session_id


    if mode == 'multi':
        chat_history = [m.model_dump() for m in query_request.chat_history]
    else:
        chat_history = None
    
    def token_generator():

        for frame in stream_answer_question(
                                        question = question,
                                        chat_history = chat_history,
                                        session_id = session_id):
            
            if frame['type'] == 'sources':
                sources = frame['sources']
                sources_list = [dict(
                                    content = d.page_content,
                                    source = d.metadata['source'],
                                    page_number = d.metadata.get('page_number'))
                                for d in sources    
                                ]
                yield json.dumps({'type': 'sources', 'sources': sources_list}) + '\n'

            else:
                yield json.dumps({'type': 'token', 'token': frame['token']}) + '\n'

        yield json.dumps({'type': 'done'}) + '\n'

    return StreamingResponse(token_generator(), media_type = 'application/x-ndjson')


@router.delete('/delete_single_file', response_model = DeleteResponse)
def delete_file(filename: str = Body(...)):

    deletion_status = delete_single_file(filename = filename)
    deleted, message, chunks_deleted = deletion_status['deleted'], deletion_status['message'], deletion_status['chunks_deleted']

    if not deleted:
        raise HTTPException(status_code = 400, detail = f"File '{filename}' doesn't exist. Try again.")

    return DeleteResponse(
                            message = message,
                            chunk_count = chunks_deleted,
                            filename_or_all = filename
                         )


@router.delete('/delete_all_files', response_model = DeleteResponse)
def delete_all():

    deletion_status = delete_all_files()
    deleted, message, chunks_deleted = deletion_status['deleted'], deletion_status['message'], deletion_status['chunks_deleted']

    return DeleteResponse(
                            message = message,
                            chunk_count = chunks_deleted,
                            filename_or_all = 'ALL'
                         )


if __name__ == '__main__':
    pass