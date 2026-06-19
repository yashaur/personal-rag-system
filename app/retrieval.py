from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from app.vectorstore import vectorstore
from app.config import settings

TOP_K = settings.top_k
RERANK_TOP_N = settings.rerank_top_n
RERANKER_MODEL = settings.reranker_model
WEIGHTS = settings.hybrid_weights

_semantic_retriever = vectorstore.as_retriever(
                                            search_kwargs = {
                                                            'k': TOP_K
                                                            }
                                                )

_reranker = CrossEncoderReranker(
                                model = HuggingFaceCrossEncoder(model_name = RERANKER_MODEL),
                                top_n = RERANK_TOP_N
                                )

def _all_docs_from_chroma() -> list[Document]:
    data = vectorstore.get(include = ['documents', 'metadatas'])
    return [Document(page_content = t, metadata = m) for t, m in zip(data['documents'], data['metadatas'])]

def _build_retriever() -> ContextualCompressionRetriever | None:
    docs = _all_docs_from_chroma()

    if not docs:
        return None
    
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = TOP_K

    ensemble_retriever = EnsembleRetriever(
                                            retrievers = [_semantic_retriever, bm25],
                                            weights = WEIGHTS
    )

    final_retriever_post_compression = ContextualCompressionRetriever(
                                                                        base_compressor = _reranker,
                                                                        base_retriever = ensemble_retriever
                                                                    )
    
    return final_retriever_post_compression


retriever = _build_retriever()

def refresh() -> None:
    global retriever
    retriever = _build_retriever()


if __name__ == '__main__':

    from app.ingestion_deletion import ingest_file

    path = settings.uploads_dir + '/test.pdf'

    vectorstore.reset_collection()

    docs = ingest_file(path)

    refresh()

    response = retriever.invoke("What is multicollinearity?")

    print(response)

    print(len(response))

    for doc in response:
        print(doc.page_content)

    vectorstore.reset_collection()