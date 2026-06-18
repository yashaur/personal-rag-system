from app.config import settings
from app.schemas import (
                            QueryRequest,
                            QueryResponse,
                            SourceDocument,
                            IngestResponse,
                            DocumentInfo
                        )
from app.ingestion import ingest_file
from app.chains import answer_question, stream_answer_question
from app.vectorstore import vectorstore
import app.retrieval as retrieval

from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from fastapi.responses import StreamingResponse
from pathlib import Path
from collections import Counter
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

    if mode == 'multi':
        chat_history = [m.model_dump() for m in query_request.chat_history]
    else:
        chat_history = None

    response = answer_question(question = question, chat_history = chat_history)

    sources = [SourceDocument(
                            content = doc.page_content,
                            source = doc.metadata['source'],
                            page_number = doc.metadata.get('page_number'))
                for doc in response['sources']
            ]
    
    return QueryResponse(answer = response['answer'], sources = sources)


@router.get('/documents', response_model = list[DocumentInfo])
def retrieve_docs():
    filedata = vectorstore.get(include = ['metadatas'])
    filedata_with_n_chunks = Counter(m['source'] for m in filedata['metadatas'])
    return [DocumentInfo(filename=fname, chunk_count=filedata_with_n_chunks[fname]) for fname in filedata_with_n_chunks]


@router.post('/query/stream')
def query_stream(query_request: QueryRequest = Body(...)):
    question = query_request.question
    mode = query_request.mode

    if mode == 'multi':
        chat_history = [m.model_dump() for m in query_request.chat_history]
    else:
        chat_history = None

    sources_doc, token_iterator = stream_answer_question(question = question, chat_history = chat_history)

    sources = [        dict(
                            content = doc.page_content,
                            source = doc.metadata['source'],
                            page_number = doc.metadata.get('page_number')
                            )
                for doc in sources_doc
                ]
    
    def token_generator():

        header = json.dumps({'type': 'sources', 'sources': sources}) + '\n'

        footer = json.dumps({'type': 'done'}) + '\n'

        yield header

        for token in token_iterator:
            token_str = json.dumps({'type': 'token', 'token': token}) + '\n'
            yield token_str
        
        yield footer

    return StreamingResponse(token_generator(), media_type = 'application/x-ndjson')