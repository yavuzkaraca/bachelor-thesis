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


def combined_all(llm, doc):
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
        (
            "system", system_default_role() + output_format()
        ),
        ("user",
         instructions_chain_of_thought() + first_response.content + completeness_types() + examples()
         + "\n".join([page.page_content for page in doc])
         + instructions_chain_of_thought() + first_response.content + completeness_types() + examples()),
    ]

    response = with_message_history.invoke(messages, config=config)

    return response.content


def combined_cot_ri(llm, doc):
    """
    Combines:
     1. chain of though
     2. repeated instructions
     3. few shot

    """
    messages = [
        (
            "system", system_default_role() + output_format()
        ),
        ("user", instructions_chain_of_thought() + examples() + "\n".join(
            [page.page_content for page in doc])
         + "\n" + instructions_chain_of_thought() + examples()),
    ]
    response = llm.invoke(messages)
    return response.content


def combined_gk_types(llm, doc):
    """
    Combines:
     1. few shot
     2. generated knowledge (gk)
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
            "system", system_default_role() + output_format()
        ),
        ("user", instructions_base() + examples() + first_response.content + completeness_types()
         + "\n".join([page.page_content for page in doc])),
    ]

    response = with_message_history.invoke(messages, config=config)

    return response.content


def generated_knowledge(llm, doc):
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
        ("user", instructions_base() + examples() + "\n".join([page.page_content for page in doc])),
    ]

    response = with_message_history.invoke(messages, config=config)

    return response.content


def few_shot(llm, doc):
    messages = [
        (
            "system", system_default_role() + output_format()
        ),
        ("user", instructions_base() + examples() + "\n".join([page.page_content for page in doc])),
    ]

    response = llm.invoke(messages)
    return response.content


def zero_shot(llm, doc):
    messages = [
        (
            "system", system_default_role() + output_format()
        ),
        ("user", instructions_base() + "\n".join([page.page_content for page in doc])),
    ]

    response = llm.invoke(messages)
    return response.content


def engineer_persona(llm, doc):
    messages = [
        (
            "system", system_engineer_role() + output_format()
        ),
        ("user", instructions_base() + examples() + "\n".join([page.page_content for page in doc])),
    ]

    response = llm.invoke(messages)
    return response.content


def repeated_instructions(llm, doc):
    messages = [
        (
            "system", system_default_role() + output_format()
        ),
        ("user", instructions_base() + examples() + "\n".join([page.page_content for page in doc])
         + "\n" + instructions_base() + examples())
    ]

    response = llm.invoke(messages)
    return response.content


def chain_of_thought_zero_shot(llm, doc):
    messages = [
        (
            "system", system_default_role() + output_format()
        ),
        ("user", instructions_chain_of_thought() + "\n".join([page.page_content for page in doc])),
    ]

    response = llm.invoke(messages)
    return response.content


def chain_of_thought_few_shot(llm, doc):
    messages = [
        (
            "system", system_default_role() + output_format()
        ),
        ("user", instructions_chain_of_thought() + examples() + "\n".join(
            [page.page_content for page in doc])),
    ]

    response = llm.invoke(messages)
    return response.content


def provided_completeness_types(llm, doc):
    messages = [
        (
            "system", system_default_role() + output_format()
        ),
        ("user", instructions_base() + completeness_types() + "\n".join(
            [page.page_content for page in doc])),
    ]

    response = llm.invoke(messages)
    return response.content
