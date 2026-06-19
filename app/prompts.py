from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

init_messages = [
    ('system', "You are a helpful RAG chatbot and assistant. Your job is to help the user answer any questions they might have strictly from the context provided. Anything being asked outside the context must not be answered, simply let the user know that the context doesn't cover this specific portion of the query. When you state specific information, cite the source filename it came from, as shown in the context. Remember to create responses that are not too verbose and use a professional, neutral tone and avoid using any emojis or emoticons."),
    ('user', "Question:\n{question}\n\nContext:\n{context}")
]

rag_prompt = ChatPromptTemplate.from_messages(init_messages)

system_message = """
Given the following conversation between the user and the assistant and the final question by the user,
produce a standalone question that is fully understandable by the assistance without the needing to parse the chat history.
You MUST NOT answer the user's final question – ONLY REFORMULATE or CONDENSE it.
If the follow-up is already standalone and does not require any reformulation from the chat history, return the question unchanged.
Output ONLY the rewritten/condensed question, no preamble, no quotes, and no "Standalone question:" prefixes.
"""


multi_turn_messages = [
    ('system', system_message),
    MessagesPlaceholder('chat_history'),
    ('user', 'Final question: {question}')
]

condenser_prompt = ChatPromptTemplate.from_messages(multi_turn_messages)


if __name__ == '__main__':
    # for message in (condenser_prompt.messages):
    #     print('\n', message, '\n')
    
    print(condenser_prompt.invoke({'question': 'Hi i have a question', 'chat_history': [('user', "Hi"), ('assistant', 'Hi')]}))