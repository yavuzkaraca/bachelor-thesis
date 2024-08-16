"""
This module provides functions to invoke a language model (LLM) with different prompts for document validation.

Functions:
    - validate_with_full_prompt: Validates documents using a prompt containing instructions, IEEE guidelines, and completeness types.
    - validate_without_completeness_types: Validates documents using a prompt containing instructions and IEEE guidelines, excluding completeness types.
    - validate_with_instructions_only: Validates documents using a prompt containing only instructions.
"""

from llm.prompts import ieee_guidelines, instructions_few_shot, completeness_types, system_default_role, \
    instructions_zero_shot, system_engineer_role, instructions_chain_of_thought


def generated_knowledge_all(llm, docs):
    """
    Validates documents using a comprehensive prompt combining instructions, IEEE guidelines, and completeness types.

    Args:
        llm: The language model instance to invoke.
        docs: A list of document objects to validate.

    Returns:
        The content of the LLM's response.
    """
    messages = [
        (
            "system", system_default_role()
        ),
        ("user", instructions_few_shot() + ieee_guidelines() + completeness_types()
         + "\n".join([doc.page_content for doc in docs])),
    ]

    response = llm.invoke(messages)
    return response.content


def generated_knowledge_ieee(llm, docs):
    """
    Validates documents using a prompt that includes instructions and IEEE guidelines, but excludes completeness types.

    Args:
        llm: The language model instance to invoke.
        docs: A list of document objects to validate.

    Returns:
        The content of the LLM's response.
    """
    messages = [
        (
            "system", system_default_role()
        ),
        ("user", instructions_few_shot() + ieee_guidelines() + "\n".join([doc.page_content for doc in docs])),
    ]

    response = llm.invoke(messages)
    return response.content


def few_shot(llm, docs):
    """
    Validates documents using a minimal prompt containing only instructions.

    Args:
        llm: The language model instance to invoke.
        docs: A list of document objects to validate.

    Returns:
        The content of the LLM's response.
    """

    messages = [
        (
            "system", system_default_role()
        ),
        ("user", instructions_few_shot() + "\n".join([doc.page_content for doc in docs])),
    ]

    response = llm.invoke(messages)
    return response.content


def zero_shot(llm, docs):
    """
    Validates documents using a minimal prompt containing only instructions.

    Args:
        llm: The language model instance to invoke.
        docs: A list of document objects to validate.

    Returns:
        The content of the LLM's response.
    """

    messages = [
        (
            "system", system_default_role()
        ),
        ("user", instructions_zero_shot() + "\n".join([doc.page_content for doc in docs])),
    ]

    response = llm.invoke(messages)
    return response.content


def engineer_persona(llm, docs):
    """
    Validates documents using a minimal prompt containing only instructions.

    Args:
        llm: The language model instance to invoke.
        docs: A list of document objects to validate.

    Returns:
        The content of the LLM's response.
    """

    messages = [
        (
            "system", system_engineer_role()
        ),
        ("user", instructions_few_shot() + "\n".join([doc.page_content for doc in docs])),
    ]

    response = llm.invoke(messages)
    return response.content


def repeated_instructions(llm, docs):
    """
    Validates documents using a minimal prompt containing only instructions.

    Args:
        llm: The language model instance to invoke.
        docs: A list of document objects to validate.

    Returns:
        The content of the LLM's response.
    """

    messages = [
        (
            "system", system_engineer_role()
        ),
        ("user", instructions_few_shot() + "\n".join([doc.page_content for doc in docs])
         + "\n" + instructions_few_shot()),
    ]

    response = llm.invoke(messages)
    return response.content


def chain_of_thought(llm, docs):
    """
    Validates documents using a minimal prompt containing only instructions.

    Args:
        llm: The language model instance to invoke.
        docs: A list of document objects to validate.

    Returns:
        The content of the LLM's response.
    """

    messages = [
        (
            "system", system_default_role()
        ),
        ("user", instructions_chain_of_thought() + "\n".join([doc.page_content for doc in docs])),
    ]

    response = llm.invoke(messages)
    return response.content
