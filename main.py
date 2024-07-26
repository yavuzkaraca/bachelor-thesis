import os
from langchain_openai import ChatOpenAI

# Securely setting up the OpenAI API key
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


def basic_test():
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=OPENAI_API_KEY
        # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
        # base_url="...",
        # organization="...",
        # other params...
    )

    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to German. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]
    ai_msg = llm.invoke(messages)

    print(ai_msg)

    print(ai_msg.content)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    basic_test()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
