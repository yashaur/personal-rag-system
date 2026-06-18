from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

from app.llm import llm
from app.prompts import rag_prompt, condenser_prompt
import app.retrieval as retrieval

from time import perf_counter
import logging

from collections.abc import Iterator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

parser = StrOutputParser()

def add_timer(original_chain, chain_name: str):
    def timed(chain_input):
        start = perf_counter()
        result = original_chain.invoke(chain_input)
        elapsed = perf_counter() - start
        logger.info("%s: %.3fs", chain_name + ' chain', elapsed)
        return result
    return RunnableLambda(timed)

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

timed_retrieval_chain = add_timer(retrieval_chain, 'Retrieval + reranking')

context_dict = {
                    'question': lambda d: d['question'],
                    'context': lambda d: format_chunks(d['retrieved_chunks'])
                                }

generation_chain = rag_prompt | llm | parser

timed_generation_chain = add_timer(generation_chain, 'LLM generation')

answer_chain = RunnablePassthrough.assign(
                    answer = context_dict | timed_generation_chain
                                        )

final_output_chain = RunnableLambda(
                        lambda d: {'answer': d['answer'], 'sources': d['retrieved_chunks']}
)

single_turn_chain = timed_retrieval_chain | answer_chain | final_output_chain


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

condense_chain = condenser_prompt | llm | parser

timed_condense_chain = add_timer(condense_chain, 'Condenser')

def standalone_question(query: dict) -> str:
    history = query.get('chat_history')

    if not history:
        return query['question']
    
    return timed_condense_chain.invoke({
            'question': query['question'],
            'chat_history': to_messages(history)
                                })

standalone_chain = RunnablePassthrough.assign(
                                            question = RunnableLambda(standalone_question)
)

rag_chain_final = standalone_chain | single_turn_chain


def answer_question(question: str, chat_history: list[dict] | None = None) -> dict:
    if retrieval.retriever is None:
        return {'answer': 'No documents have been ingested yet!',
                'sources': []
                }
    
    return rag_chain_final.invoke({
                                'question': question,
                                'chat_history': chat_history or []
    })


### STREAMING CHAIN ###

stream_generation_chain = llm | parser

def stream_answer_question(question: str, chat_history: list[dict] | None = None) -> tuple[list[Document], Iterator[str]]:

    if retrieval.retriever is None:
        return (
                [], # empty sources list for the API because nothing was ingested
                iter(['No documents have been ingested yet!']) # Trying to keep the return object of this function consistent as a tuple of list of sources and an iterator
        )

    condense_retrieval_chain = standalone_chain | timed_retrieval_chain

    retrieval_dict = condense_retrieval_chain.invoke({
                                                'question': question,
                                                'chat_history': chat_history or []
                                                    })
    
    sources = retrieval_dict['retrieved_chunks']

    prompt_chain = context_dict | rag_prompt

    prompt = prompt_chain.invoke(retrieval_dict)

    print("Chat history so far:\n", chat_history)
    print("Prompt being sent to LLM:\n", prompt)

    return sources, stream_generation_chain.stream(prompt)


if __name__ == '__main__':

    from app.ingestion import ingest_file
    from app.config import settings

    retrieval.vectorstore.reset_collection()
    path = settings.uploads_dir + '/test.pdf'
    ingest_file(path)
    retrieval.refresh()

    multi_question = input("Do you want to turn on multi-turn questions? (Y/n): ")

    if multi_question != 'Y':
        query = input("ask a question: ")
        response = answer_question(query)
        for k in response:
            print(f'{k.title()}:\n{response[k]}\n')

        print(type(response['sources'][0]))

    else:
        chat_history = []
        while True:
            query = input("ask a question (type 'exit' to end): ")
            if query == 'exit':
                break
            response = answer_question(query, chat_history)['answer']
            print(f'assistant: {response}')

            chat_history.append({'role': 'user', 'content': query})
            chat_history.append({'role': 'assistant', 'content': response})