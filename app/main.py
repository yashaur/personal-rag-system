from fastapi import FastAPI
from app.api import router

app = FastAPI(title = 'Personal RAG System')

app.include_router(router)