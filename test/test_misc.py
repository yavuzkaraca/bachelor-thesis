from src.llm import llm_creator
from src.utils.dataset_loader import paginate_pdf

test_path_esa = "../dataset/PURE/2001_esa.pdf"
test_path_neutero = "../dataset/PSE/2022_neutero.pdf"
test_path_home = "../dataset/PURE/2010_home_1.3.pdf"


def test_chat_memory():
    """
    This test shows that each invoke is a separate new chat
    """
    llm = llm_creator.create_llm_openai()

    first_message = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]
    first_answer = llm.invoke(first_message)
    print(first_answer.content)

    second_message = [{"role": "user", "content": "Do you remember your task?"}]
    second_answer = llm.invoke(second_message)
    print(second_answer.content)


def test_text_conversion():
    """
    This test shows that all the structures are lost when pdfs are loaded
    """
    pages = paginate_pdf(test_path_esa)

    print("\nPAGE 6")
    print(pages[6])


def test_generated_ieee():
    """
    To see if model generates the same guidelines
    """
    llm = llm_creator.create_llm_openai()

    message = [{"role": "user", "content": "Provide the IEEE guidelines for Software Requirement Specification"
                                           "documents about the Completeness aspect. According to IEEE, documents"
                                           "should fulfill three things to be complete. What are these?"}]
    answer = llm.invoke(message)

    print(answer.content)


if __name__ == '__main__':
    test_chat_memory()
    test_generated_ieee()
    test_text_conversion()
