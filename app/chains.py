from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

from langfuse import get_client

from app.llm import llm, condenser_llm
from app.prompts import rag_prompt, condenser_prompt
import app.retrieval as retrieval
from app.observability import get_langfuse_handler, OllamaLatencyCallback

from collections.abc import Iterator

parser = StrOutputParser()

def format_chunks(chunks: list[Document]) -> str:
    blocks = []
    for chunk in chunks:
        source = chunk.metadata['source']
        page = chunk.metadata.get('page_number')
        
        source_str = f'Source: {source}, page: {page}' if page is not None else f'Source: {source}'

        context = ' '.join(chunk.page_content.split())

        blocks.append(source_str + '\n' + context)

    return '\n\n'.join(blocks)

### SINGLE-TURN CHAIN ###

# def remove_asterisk(text: str) -> str:
#     return text.replace('**', '')

# format_output = RunnableLambda(remove_asterisk)

retrieval_chain = RunnablePassthrough.assign(
                    retrieved_chunks = lambda x: retrieval.retriever.invoke(x['question'])
                                        )

context_dict = {
                    'question': lambda d: d['question'],
                    'context': lambda d: format_chunks(d['retrieved_chunks'])
                                }

generation_chain = rag_prompt | llm | parser

answer_chain = RunnablePassthrough.assign(
                    answer = context_dict | generation_chain
                                        )

final_output_chain = RunnableLambda(
                        lambda d: {'answer': d['answer'], 'sources': d['retrieved_chunks']}
)

single_turn_chain = retrieval_chain | answer_chain | final_output_chain


### MULTI-TURN CHAIN ###

def to_messages(chat_history: list[dict]) -> list:
    messages = []
    for message in chat_history:
        role = message['role']

        if role == 'user':
            messages.append(HumanMessage(content = message['content']))
        elif role == 'assistant':
            messages.append(AIMessage(content = message['content']))
        else:
            pass
            # To add more elif statements if there are more types of messages added in future
    
    return messages

condense_chain = condenser_prompt | condenser_llm | parser

def standalone_question(query: dict) -> str:
    history = query.get('chat_history')

    if not history:
        return query['question']
    
    history = history[-6:]
    
    rewritten =  condense_chain.invoke({
            'question': query['question'],
            'chat_history': to_messages(history)
                                }).strip()
    
    return rewritten or query['question']

standalone_chain = RunnablePassthrough.assign(
                                            question = RunnableLambda(standalone_question)
)

rag_chain_final = standalone_chain | single_turn_chain


def answer_question(question: str, chat_history: list[dict] | None = None, session_id: str | None = None) -> dict:
    if retrieval.retriever is None:
        return {'answer': 'No documents have been ingested yet!',
                'sources': []
                }
    
    client = get_client()
    trace_id = client.create_trace_id()
    handler = get_langfuse_handler(trace_id = trace_id)

    lf_meta = {
                'langfuse_trace_name': 'non-stream-request',
                'langfuse_user_id': 'yashaur',
                'langfuse_tags': ['live', 'stream:off'],
            }
    if session_id:
        lf_meta['langfuse_session_id'] = session_id

    standalone_chain_output = standalone_chain.invoke({
                                                    'question': question,
                                                    'chat_history': chat_history or []
                                                    },
                                                    config = {'callbacks': [handler, OllamaLatencyCallback(label = 'condense', trace_id = trace_id)],
                                                              'metadata': lf_meta,
                                                              'run_name': 'condense'}
                                                        )

    final_output = single_turn_chain.invoke(
                        standalone_chain_output,
                        config = {'callbacks': [handler, OllamaLatencyCallback(label = 'answer', trace_id = trace_id)],
                                  'metadata': lf_meta,
                                  'run_name': 'answer'}
                        )

    return final_output


### STREAMING CHAIN ###

stream_generation_chain = llm | parser

def stream_answer_question(question: str, chat_history: list[dict] | None = None, session_id: str | None = None) -> Iterator:

    if retrieval.retriever is None:
        yield {'type': 'sources', 'sources': []} # empty sources list for the API because nothing was ingested
        yield {'type': 'token', 'token': 'No documents have been ingested yet!'} # Trying to keep the return object of this function consistent as a tuple of list of sources and an iterator
        return
    
    client = get_client()
    trace_id = client.create_trace_id()
    handler = get_langfuse_handler(trace_id = trace_id)

    lf_meta = {
                'langfuse_trace_name': 'stream-request',
                'langfuse_user_id': 'yashaur',
                'langfuse_tags': ['live', 'stream:on'],
            }
    if session_id:
        lf_meta['langfuse_session_id'] = session_id

    condense_retrieval_chain = standalone_chain | retrieval_chain

    retrieval_dict = condense_retrieval_chain.invoke({
                                                'question': question,
                                                'chat_history': chat_history or []
                                                    },
                                                config = {'callbacks': [handler, OllamaLatencyCallback(label = 'condense', trace_id = trace_id)],
                                                          'metadata': lf_meta,
                                                          'run_name': 'condense'}
                                                )

    sources = retrieval_dict['retrieved_chunks']

    yield {'type': 'sources', 'sources': sources}

    prompt_chain = context_dict | rag_prompt

    prompt = prompt_chain.invoke(retrieval_dict)

    stream_gen_chain = stream_generation_chain.stream(
                                        prompt,
                                        config = {'callbacks': [handler, OllamaLatencyCallback(label = 'answer', trace_id = trace_id)],
                                                  'metadata': lf_meta,
                                                  'run_name': 'answer'})

    for chunk in stream_gen_chain:
        yield {'type': 'token', 'token': chunk}


if __name__ == '__main__':

    # from app.ingestion import ingest_file
    # from app.config import settings

    # retrieval.vectorstore.reset_collection()
    # path = settings.uploads_dir + '/test.pdf'
    # ingest_file(path)
    # retrieval.refresh()

    # multi_question = input("Do you want to turn on multi-turn questions? (Y/n): ")

    # if multi_question != 'Y':
    #     query = input("ask a question: ")
    #     response = answer_question(query)
    #     for k in response:
    #         print(f'{k.title()}:\n{response[k]}\n')

    #     print(type(response['sources'][0]))

    # else:
    #     chat_history = []
    #     while True:
    #         query = input("ask a question (type 'exit' to end): ")
    #         if query == 'exit':
    #             break
    #         response = answer_question(query, chat_history)['answer']
    #         print(f'assistant: {response}')

    #         chat_history.append({'role': 'user', 'content': query})
    #         chat_history.append({'role': 'assistant', 'content': response})

    
    hist = [
    {'role': 'user', 'content': 'What is GDP'},
    {'role': 'assistant', 'content': 'GDP is a measure of aggregate output...'},
    ]
    print(standalone_question({'question': 'how can i measure it', 'chat_history': hist}))
