from templates import IEEE_guidelines, instructions, completeness_types


def analyze_pdf_incompleteness(llm, docs):
    messages = [
        {"role": "user", "content": instructions() + IEEE_guidelines() + completeness_types() + "\n".join(
            [doc.page_content for doc in docs])}
    ]

    response = llm.invoke(messages)
    return response.content
