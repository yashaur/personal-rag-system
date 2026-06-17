from app.config import settings
from app.schemas import (
                            QueryRequest,
                            QueryResponse,
                            SourceDocument,
                            IngestResponse,
                            DocumentInfo
                        )
from app.ingestion import ingest_file
from app.chains import answer_question
from app.vectorstore import vectorstore
import app.retrieval as retrieval

from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from pathlib import Path
from collections import Counter

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