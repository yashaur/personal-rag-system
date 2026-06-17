import os
import httpx
from typing import Literal

BASE_URL = os.getenv('RAG_API_URL', 'http://localhost:8000')

client = httpx.Client(base_url = BASE_URL, timeout = httpx.Timeout(300.0))

def ingest(uploaded_file) -> dict:
    file = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    response = client.post('/ingest', files = file)
    response.raise_for_status()
    return response.json()

def query(question: str, chat_history: list | None = None, mode: Literal['single', 'multi'] = 'single') -> dict:
    query_json = {'question': question, 'mode': mode, 'chat_history': chat_history}
    response = client.post('/query', json = query_json)
    response.raise_for_status()
    return response.json()

def list_documents() -> list[dict]:
    response = client.get('/documents')
    response.raise_for_status()
    return response.json()