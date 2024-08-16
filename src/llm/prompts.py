"""
This module provides template functions that generate standard prompts and guidelines for validating Software
Specification Documents.

Functions:
    - instructions: Returns a detailed prompt instructing an assistant on how to identify and report incompleteness in
    Software Specification Documents.
    - ieee_guidelines: Returns a description of the importance of completeness in Software Requirements Specifications
    (SRS) and outlines IEEE guidelines for ensuring a complete SRS document.
    - completeness_types: Returns a detailed explanation of the concept of completeness in Software Requirements
    Specifications (SRS) and outlines three levels of completeness in SRS documents.

"""


def system_engineer_role():
    return """
    You are an experienced Requirements Engineer that identifies incompleteness in Software Requirement Specification 
    documents.
    """


def system_default_role():
    return """
    You are an AI assistant that identifies incompleteness in Software Requirement Specification documents.
    """


def instructions_few_shot():
    """
    Provides a prompt for an assistant tasked with identifying incompleteness in Software Specification Documents.

    The prompt details the steps the assistant should follow:
        1. Refer to the requirement by its unique identifier, section number, or "N/A" if it's document-wide.
        2. State the issue found.
        3. Offer a correction or suggestion.

    Output Format:
        The output is expected to be a CSV file with columns: "Label", "Issue", and "Suggestion".

    Returns:
        A string containing the instructions and an example output.
    """
    return """
    Identify all the instances of incompleteness in the following document by:
    1. Referring to the requirement by its unique identifier/label. If it's more than one requirement then you can
     refer to the section number. If the section is totally missing than you can say "N/A".
    2. Stating the issue.
    3. Providing an example of a requirement that addresses the identified incompleteness.

    Output Format:
    Produce a CSV file with the following columns: "Label", "Issue", and "Correction". Ensure that each cell 
    value in the CSV file is separated with a semicolon (;).

    Example Output:
    R19;does not specify what happens if the message is ignored;If the immunization reminder is ignored, the system shall send an alert to the administrator
    Section 5.1;does not specify total supported number of concurrent users;The system shall handle up to 10.000 concurrent users without performance degradation
    FR-A-30;does not specify recovery or program state when selection is improper;Improper Selection is undone and the previous state before the selection is still valid
    \n
    """


def instructions_chain_of_thought():
    """
    Provides a prompt for an assistant tasked with identifying incompleteness in Software Specification Documents.

    The prompt details the steps the assistant should follow:
        1. Refer to the requirement by its unique identifier, section number, or "N/A" if it's document-wide.
        2. State the issue found.
        3. Offer a correction or suggestion.

    Output Format:
        The output is expected to be a CSV file with columns: "Label", "Issue", and "Suggestion".

    Returns:
        A string containing the instructions and an example output.
    """
    return """
    First, read the following document thoroughly and identify all the instances of incompleteness.
    Then explain the reason of incompleteness for each instance referring to their unique identifier.
    Finally, depending on the reason, provide an example of a requirement that addresses the identified incompleteness.

    Output Format:
    Produce a CSV file with the following columns: "Label", "Issue", and "Correction". Ensure that each cell 
    value in the CSV file is separated with a semicolon (;).
    \n
    """


def instructions_zero_shot():
    """
    Provides a prompt for an assistant tasked with identifying incompleteness in Software Specification Documents.

    The prompt details the steps the assistant should follow:
        1. Refer to the requirement by its unique identifier, section number, or "N/A" if it's document-wide.
        2. State the issue found.
        3. Offer a correction or suggestion.

    Output Format:
        The output is expected to be a CSV file with columns: "Label", "Issue", and "Suggestion".

    Returns:
        A string containing the instructions and an example output.
    """
    return """
    Identify all the instances of incompleteness in the following document by:
    1. Referring to the requirement by its unique identifier/label. If it's more than one requirement then you can
     refer to the section number. If the section is totally missing than you can say "N/A".
    2. Stating the issue.
    3. Providing an example of a requirement that addresses the identified incompleteness.

    Output Format:
    Produce a CSV file with the following columns: "Label", "Issue", and "Correction". Ensure that each cell 
    value in the CSV file is separated with a semicolon (;).
    \n
    """


def generate_ieee_guidelines():
    return """
    Provide the IEEE guidelines for Software Requirement Specification documents about the Completeness aspect. 
    According to IEEE, documents should fulfill three things to be complete. What are these?
    """

