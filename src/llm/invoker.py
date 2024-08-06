from llm.templates import ieee_guidelines, instructions, completeness_types


def analyze_pdf_incompleteness(llm, docs):
    messages = [
        {"role": "user", "content": instructions() + ieee_guidelines() + completeness_types() + "\n".join(
            [doc.page_content for doc in docs])}
    ]

    response = llm.invoke(messages)
    return response.content
