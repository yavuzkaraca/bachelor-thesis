"""
This module provides functions to invoke a language model (LLM) with different prompts.
"""
from langchain_core.messages import HumanMessage

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)

from langchain_core.runnables.history import RunnableWithMessageHistory

from llm.prompts import (generate_ieee_guidelines, instructions_base, system_default_role, system_engineer_role,
                         instructions_chain_of_thought, completeness_types, examples, output_format)


def chain_of_thought_few_shot(llm, doc) -> str:
    messages = {'system': system_default_role() + output_format(),
                'user_first': instructions_chain_of_thought() + examples(), 'user_second': ""}
    return invoke_helper(llm, doc, messages)


def chain_of_thought_zero_shot(llm, doc) -> str:
    messages = {'system': system_default_role() + output_format(),
                'user_first': instructions_chain_of_thought(), 'user_second': ""}
    return invoke_helper(llm, doc, messages)


def provided_completeness_types(llm, doc) -> str:
    messages = {'system': system_default_role() + output_format(),
                'user_first': instructions_base() + completeness_types(), 'user_second': ""}
    return invoke_helper(llm, doc, messages)


def few_shot(llm, doc) -> str:
    messages = {'system': system_default_role() + output_format(),
                'user_first': instructions_base() + examples(), 'user_second': ""}
    return invoke_helper(llm, doc, messages)


def zero_shot(llm, doc) -> str:
    messages = {'system': system_default_role() + output_format(),
                'user_first': instructions_base(), 'user_second': ""}
    return invoke_helper(llm, doc, messages)


def engineer_persona(llm, doc) -> str:
    messages = {'system': system_engineer_role() + output_format(),
                'user_first': instructions_base(), 'user_second': ""}
    return invoke_helper(llm, doc, messages)


def repeated_instructions(llm, doc) -> str:
    messages = {'system': system_default_role() + output_format(),
                'user_first': instructions_base() + examples(), 'user_second': instructions_base() + examples()}
    return invoke_helper(llm, doc, messages)


import sys
print (sys.version)

def generated_knowledge(llm, doc) -> str:
    # Default example for a Chat with Message History, which is needed for Generated Knowledge
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    with_message_history = RunnableWithMessageHistory(llm, get_session_history)
    config = {"configurable": {"session_id": "abc"}}

    first_response = with_message_history.invoke([HumanMessage(content=generate_ieee_guidelines())],
                                                 config=config)

    messages = [
        ("system", system_default_role() + output_format()),
        ("user", instructions_base() + examples() + first_response + "\n".join([page.page_content for page in doc])),
    ]
    response = with_message_history.invoke(messages, config=config)
    return response.content


def combined_chain_of_thought_repeated_instructions(llm, doc) -> str:
    """
    Combines:
     1. chain of though
     2. repeated instructions
     3. few shot
    """

    messages = {'system': system_default_role() + output_format(),
                'user_first': instructions_chain_of_thought() + examples(),
                'user_second': instructions_chain_of_thought() + examples()}
    return invoke_helper(llm, doc, messages)


def combined_all(llm, doc) -> str:
    """
    Combines:
     1. chain of though
     2. repeated instructions
     3. few shot
     4. generated knowledge
     5. completeness types
    """
    store = {}

    #
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    with_message_history = RunnableWithMessageHistory(llm, get_session_history)

    config = {"configurable": {"session_id": "abc"}}

    first_response = with_message_history.invoke([HumanMessage(content=generate_ieee_guidelines())],
                                                 config=config)

    messages = [
        ("system", system_default_role() + output_format()),
        ("user",
         instructions_chain_of_thought() + first_response.content + completeness_types() + examples()
         + "\n".join([page.page_content for page in doc])
         + instructions_chain_of_thought() + first_response.content + completeness_types() + examples()),
    ]
    response = with_message_history.invoke(messages, config=config)
    return response.content


def combined_generated_knowledge_completeness_types(llm, doc) -> str:
    """
    Combines:
     1. few shot
     2. generated knowledge
     3. completeness types
    """
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    with_message_history = RunnableWithMessageHistory(llm, get_session_history)
    config = {"configurable": {"session_id": "abc"}}

    first_response = with_message_history.invoke([HumanMessage(content=generate_ieee_guidelines())],
                                                 config=config)

    messages = [
        ("system", system_default_role() + output_format()),
        ("user", instructions_base() + examples() + first_response.content + completeness_types()
         + "\n".join([page.page_content for page in doc])),
    ]
    response = with_message_history.invoke(messages, config=config)
    return response.content


def invoke_helper(llm, doc, messages) -> str:
    msg = [
        ("system", messages['system']),
        ("user", messages['user_first'] + "\n".join([page.page_content for page in doc]) + messages['user_second']),
    ]
    response = llm.invoke(msg)
    return response.content
