from pydantic import BaseModel, Field
from typing import Literal, List


class ChatMessage(BaseModel):
    role: Literal['user', 'assistant'] = Field(..., description = 'The role of the entity that created the message. Can either be the user (input) or the LLM (assistant)')
    content: str = Field(..., description = 'The content of the message')

class QueryRequest(BaseModel):
    question: str = Field(..., description = 'The query asked by the user')
    chat_history: List[ChatMessage] = Field(default = [], description = 'The chat history between the user and assistant so far')
    session_id: str | None = Field(default = None, description = 'The session ID used for tracing')
    mode: Literal['single', 'multi'] = Field(default = 'single', description = 'The mode of conversation: can either be single (just one question asked and the chat ends) or multi-turn (user asks multiple questions in sequence)')

class SourceDocument(BaseModel):
    content: str = Field(..., description = "The content retrieved from a source document's chunk")
    source: str = Field(..., description = 'The source filename of the chunk')
    page_number: int | None = Field(default = None, description = "The page number of the chunk. Optionally None in case of .txt files or any future file formats that don't contain page numbers")

class QueryResponse(BaseModel):
    answer: str = Field(..., description = 'The response by the LLM to the user on their query')
    sources: List[SourceDocument] = Field(..., description = 'The list of attached sources that were used to augment the LLM response to the user query')

class IngestResponse(BaseModel):
    filename: str = Field(..., description = 'The filename with extension of the file that was ingested into the Chroma database') 
    chunk_count: int = Field(..., description = 'The number of chunks that were processed out of the file')

class DeleteResponse(BaseModel):
    message: str = Field(..., description = 'Message explaining if the file was deleted or not, and the reason if not deleted.')
    chunk_count: int = Field(..., description = 'The number of chunks that were deleted from the database. 0 chunks deleted indicates the delete operation was unsuccesful')
    filename_or_all: str | None = Field(default = None, description = 'Field specifying the filename of the file deleted, or simply "ALL", specifying deletion of the entire database. "None" if the file could not be deleted.')

class DocumentInfo(IngestResponse):
    filename: str = Field(..., description = 'The filename with extension of the file that was ingested into the Chroma database')
    page_count: int = Field(..., description = 'The number of pages of the document. 1 in case of a .txt file')
    chunk_count: int = Field(..., description = 'The number of chunks that were processed out of the file')


if __name__ == '__main__':
    from pydantic import ValidationError
    
    ### Checking for IngestResponse and DocumentInfo
    doc = IngestResponse(filename = 'test', chunk_count= 30)
    print(doc.filename)
    print(doc.chunk_count)

    doc = DocumentInfo(filename = 'test', chunk_count= 30)
    print(doc.filename)
    print(doc.chunk_count)


    ### Checking for ChatMessage and QueryRequest
    chat_history = [
                    ChatMessage(role = 'user', content = 'Hi, what are t-tests?'),
                    ChatMessage(role = 'assistant', content = 'A t-test is a statistical test used to determine if there is a significant difference between the means of two groups or to assess whether an estimated parameter in a regression model differs significantly from zero.'),
                    ChatMessage(role = 'user', content = 'What are t-tests and how can I use them in context of the document attached.'),
                    ChatMessage(role = 'assistant', 
                                content = """
                            In the provided context, t-tests are specifically mentioned in relation to:
                            1. Correlation Analysis: The correlation coefficient can be tested using a t-test to check if it is different from zero (test.pdf, page 23). This helps determine whether there is a statistically significant relationship between two variables.
                            2. Regression Parameter Inference: When multicollinearity (MC) is present in regression models, large standard errors result, which weaken the reliability of t-tests for individual parameter estimates. Thiscan lead to insignificant t-statistics, making it difficult to reject the null hypothesis that a coefficient equals zero (test.pdf, pages 14 and 18). The presence of multicollinearity also increases uncertainty in estimated parameters, affecting inference based on t-tests.
                              """)
                    ]

    query = QueryRequest(question = 'Tell me more about using t-tests for correlation analysis', chat_history = chat_history)

    print(query.model_dump())     


    ### Checking if validation errors work as expected
    try:
        ChatMessage(role = 'system', content = 'Hi, how can I help you?')
    except ValidationError as e:
        print(f'Rejected as expected:\n{e}')


    ### Checking if SourceDocument and QueryResponse work as expected
    from app.vectorstore import vectorstore

    doc_id = vectorstore.get(include=['metadatas'])['ids'][0]
    doc_details = vectorstore.get_by_ids([doc_id])[0]
    
    source_doc = SourceDocument(
                                content = doc_details.page_content,
                                source = doc_details.metadata['source'],
                                page_number = doc_details.metadata['page_number']
                                )

    print(
        f"""
        SourceDocument:
        Content: {source_doc.content}
        Source: {source_doc.source}
        Page number: {source_doc.page_number}\n"""
    )
    
    response = QueryResponse(answer = 'ABCD', sources = [source_doc])
    
    print(
        f"""
        QueryResponse:
        Answer: {response.answer}
        Sources: {response.sources}"""
    )


    # Checking for IngestResponse with ingest_file()
    from app.ingestion_deletion import ingest_file
    from app.config import settings

    path = settings.uploads_dir + '/test.pdf'
    ingest_response = IngestResponse(**ingest_file(path))

    print(f"""
        Filename: {ingest_response.filename}
        Chunk count: {ingest_response.chunk_count}""")

    