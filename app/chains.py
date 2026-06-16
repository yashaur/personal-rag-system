from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

from app.llm import llm
from app.prompts import rag_prompt, condenser_prompt
import app.retrieval as retrieval

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

def remove_asterisk(text: str) -> str:
    return text.replace('**', '')

format_output = RunnableLambda(remove_asterisk)

chunks_chain = RunnablePassthrough.assign(
                    chunks = lambda x: retrieval.retriever.invoke(x['question'])
                                        )

context_dict = {
                    'question': lambda d: d['question'],
                    'context': lambda d: format_chunks(d['chunks'])
                                }

generation_chain = rag_prompt | llm | parser | format_output

answer_chain = RunnablePassthrough.assign(
                    answer = context_dict | generation_chain
                                        )

final_output_chain = RunnableLambda(
                        lambda d: {'answer': d['answer'], 'sources': d['chunks']}
)

single_turn_chain = chunks_chain | answer_chain | final_output_chain


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

def standalone_question(query: dict) -> str:
    history = query.get('chat_history')

    if not history:
        return query['question']
    
    return condense_chain.invoke({
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