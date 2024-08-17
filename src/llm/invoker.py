"""
This module provides functions to invoke a language model (LLM) with different prompts for document validation.

Functions:
    - validate_with_full_prompt: Validates documents using a prompt containing instructions, IEEE guidelines, and completeness types.
    - validate_without_completeness_types: Validates documents using a prompt containing instructions and IEEE guidelines, excluding completeness types.
    - validate_with_instructions_only: Validates documents using a prompt containing only instructions.
"""
from langchain_core.messages import HumanMessage

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)

from langchain_core.runnables.history import RunnableWithMessageHistory

from llm.prompts import generate_ieee_guidelines, instructions_few_shot, system_default_role, \
    instructions_zero_shot, system_engineer_role, instructions_chain_of_thought, completeness_types


def combined(llm, docs):
    """
    Combines:
     1. chain of though
     2. repeated instructions
     3. persona pattern
     4. zero shots
     5. provided completeness types

    """
    messages = [
        (
            "system", system_engineer_role()
        ),
        ("user", instructions_chain_of_thought() + completeness_types() + "\n".join([doc.page_content for doc in docs])
         + "\n" + instructions_chain_of_thought()),
    ]
    response = llm.invoke(messages)
    return response.content


def generated_knowledge(llm, docs):
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
        ("system", system_default_role()),
        ("user", instructions_few_shot() + "\n".join([doc.page_content for doc in docs])),
    ]

    response = with_message_history.invoke(messages, config=config)

    return response.content


def few_shot(llm, docs):
    messages = [
        (
            "system", system_default_role()
        ),
        ("user", instructions_few_shot() + "\n".join([doc.page_content for doc in docs])),
    ]

    response = llm.invoke(messages)
    return response.content


def zero_shot(llm, docs):
    messages = [
        (
            "system", system_default_role()
        ),
        ("user", instructions_zero_shot() + "\n".join([doc.page_content for doc in docs])),
    ]

    response = llm.invoke(messages)
    return response.content


def engineer_persona(llm, docs):
    messages = [
        (
            "system", system_engineer_role()
        ),
        ("user", instructions_few_shot() + "\n".join([doc.page_content for doc in docs])),
    ]

    response = llm.invoke(messages)
    return response.content


def repeated_instructions(llm, docs):
    messages = [
        (
            "system", system_default_role()
        ),
        ("user", instructions_few_shot() + "\n".join([doc.page_content for doc in docs])
         + "\n" + instructions_few_shot()),
    ]

    response = llm.invoke(messages)
    return response.content


def chain_of_thought(llm, docs):
    messages = [
        (
            "system", system_default_role()
        ),
        ("user", instructions_chain_of_thought() + "\n".join([doc.page_content for doc in docs])),
    ]

    response = llm.invoke(messages)
    return response.content


def provided_completeness_types(llm, docs):
    messages = [
        (
            "system", system_default_role()
        ),
        ("user", instructions_few_shot() + completeness_types() + "\n".join([doc.page_content for doc in docs])),
    ]

    response = llm.invoke(messages)
    return response.content
