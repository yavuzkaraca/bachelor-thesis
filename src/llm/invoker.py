"""
This module provides functions to invoke a language model (LLM) with different prompts for document validation.

Functions:
    - validate_with_full_prompt: Validates documents using a prompt containing instructions, IEEE guidelines, and completeness types.
    - validate_without_completeness_types: Validates documents using a prompt containing instructions and IEEE guidelines, excluding completeness types.
    - validate_with_instructions_only: Validates documents using a prompt containing only instructions.
"""


from llm.prompts import ieee_guidelines, instructions, completeness_types


def validate_full_prompt(llm, docs):
    """
    Validates documents using a comprehensive prompt combining instructions, IEEE guidelines, and completeness types.

    Args:
        llm: The language model instance to invoke.
        docs: A list of document objects to validate.

    Returns:
        The content of the LLM's response.
    """
    """messages = [
        {"role": "user", "content": instructions() + ieee_guidelines() + completeness_types() + "\n".join(
            [doc.page_content for doc in docs])}
    ]"""

    messages = [
        (
            "system",
            instructions() + ieee_guidelines() + completeness_types(),
        ),
        ("human", "\n".join([doc.page_content for doc in docs])),
    ]

    response = llm.invoke(messages)
    return response.content


def validate_instructions_ieee(llm, docs):
    """
    Validates documents using a prompt that includes instructions and IEEE guidelines, but excludes completeness types.

    Args:
        llm: The language model instance to invoke.
        docs: A list of document objects to validate.

    Returns:
        The content of the LLM's response.
    """
    messages = [
        {"role": "user", "content": instructions() + ieee_guidelines() + "\n".join(
            [doc.page_content for doc in docs])}
    ]

    response = llm.invoke(messages)
    return response.content


def validate_instructions_only(llm, docs):
    """
    Validates documents using a minimal prompt containing only instructions.

    Args:
        llm: The language model instance to invoke.
        docs: A list of document objects to validate.

    Returns:
        The content of the LLM's response.
    """

    """messages = [
        {"role": "user", "content": instructions() + "\n".join(
            [doc.page_content for doc in docs])}
    ]"""

    messages = [
        (
            "system",
            instructions(),
        ),
        ("human", "\n".join([doc.page_content for doc in docs])),
    ]

    response = llm.invoke(messages)
    return response.content
