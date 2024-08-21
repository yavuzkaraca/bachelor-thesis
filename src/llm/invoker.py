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

from llm.prompts import (generate_ieee_guidelines, instructions_simple, system_default_role, system_engineer_role,
                         instructions_chain_of_thought, completeness_types, examples, output_format)


def combined_all(llm, docs):
    """
    Combines:
     1. chain of though
     2. repeated instructions
     3. few shot
     4. generated knowledge
     5. completeness types

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
        (
            "system", system_default_role()
        ),
        ("user",
         instructions_chain_of_thought() + output_format() + first_response.content + completeness_types() + examples()
         + "\n".join([doc.page_content for doc in docs])
         + instructions_chain_of_thought() + output_format() + first_response.content + completeness_types() + examples()),
    ]

    response = with_message_history.invoke(messages, config=config)

    return response.content


def combined_cot_ri(llm, docs):
    """
    Combines:
     1. chain of though
     2. repeated instructions
     3. few shot

    """
    messages = [
        (
            "system", system_default_role()
        ),
        ("user", instructions_chain_of_thought() + output_format() + examples() + "\n".join(
            [doc.page_content for doc in docs])
         + "\n" + instructions_chain_of_thought() + output_format() + examples()),
    ]
    response = llm.invoke(messages)
    return response.content


def combined_gk_types(llm, docs):
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
        (
            "system", system_default_role()
        ),
        ("user", instructions_simple() + output_format() + first_response.content + completeness_types() + examples()
         + "\n".join([doc.page_content for doc in docs])),
    ]

    response = with_message_history.invoke(messages, config=config)

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
        ("user", instructions_simple() + output_format() + examples() + "\n".join([doc.page_content for doc in docs])),
    ]

    response = with_message_history.invoke(messages, config=config)

    return response.content


def few_shot(llm, docs):
    messages = [
        (
            "system", system_default_role()
        ),
        ("user", instructions_simple() + output_format() + examples() + "\n".join([doc.page_content for doc in docs])),
    ]

    response = llm.invoke(messages)
    return response.content


def zero_shot(llm, docs):
    messages = [
        (
            "system", system_default_role()
        ),
        ("user", instructions_simple() + output_format() + "\n".join([doc.page_content for doc in docs])),
    ]

    response = llm.invoke(messages)
    return response.content


def engineer_persona(llm, docs):
    messages = [
        (
            "system", system_engineer_role()
        ),
        ("user", instructions_simple() + output_format() + examples() + "\n".join([doc.page_content for doc in docs])),
    ]

    response = llm.invoke(messages)
    return response.content


def repeated_instructions(llm, docs):
    messages = [
        (
            "system", system_default_role()
        ),
        ("user", instructions_simple() + output_format() + examples() + "\n".join([doc.page_content for doc in docs])
         + "\n" + instructions_simple() + output_format() + examples())
    ]

    response = llm.invoke(messages)
    return response.content


def chain_of_thought_zero_shot(llm, docs):
    messages = [
        (
            "system", system_default_role()
        ),
        ("user", instructions_chain_of_thought() + output_format() + "\n".join([doc.page_content for doc in docs])),
    ]

    response = llm.invoke(messages)
    return response.content


def chain_of_thought_few_shot(llm, docs):
    messages = [
        (
            "system", system_default_role()
        ),
        ("user", instructions_chain_of_thought() + output_format() + examples() + "\n".join(
            [doc.page_content for doc in docs])),
    ]

    response = llm.invoke(messages)
    return response.content


def provided_completeness_types(llm, docs):
    messages = [
        (
            "system", system_default_role()
        ),
        ("user", instructions_simple() + output_format() + completeness_types() + "\n".join(
            [doc.page_content for doc in docs])),
    ]

    response = llm.invoke(messages)
    return response.content
