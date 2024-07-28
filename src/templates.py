def prepare_llm(llm):
    preparation_message = {
        "role": "system",
        "content": "You are an assistant that identifies incompleteness in Software Specification Documents."
                   "For this you will follow the guidelines:"
                   + guidelines()
    }
    return llm.invoke([preparation_message])


def analyze_pdf_incompleteness(llm, docs):
    messages = [
        {"role": "user", "content": "Analyze the following sections for incompleteness:\n" + "\n".join(
            [doc.page_content for doc in docs])}
    ]

    response = llm.invoke(messages)
    return response.content


def guidelines():
    """
    Returns a string that describes the importance of completeness and the consequences of incompleteness
    in Software Requirements Specifications (SRS), along with IEEE guidelines for a complete SRS document.
    """
    return """
    Incompleteness "causes uncertainty of the project foundations", which forces
    engineers to make assumptions when dealing with incomplete requirements. "Poor
    and incomplete SRS and inadequate requirements management are among the main
    reasons for project failure", therefore validating SRS completeness is vital.
    According to IEEE, an SRS document is complete if it:
    1. Includes all significant requirements, such as functionality, performance, design
       constraints, attributes, or external interfaces.
    2. Defines responses to all realizable input data in any conceivable situation.
    3. Provides complete labels and references for all figures, tables, and diagrams, and
       defines all terms and units of measure.
    """
