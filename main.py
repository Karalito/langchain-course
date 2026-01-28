import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course!")

    information = """ 
    Robert Greene (born May 14, 1959) is an American author of books on strategy, power, and seduction.[1] He has written seven international bestsellers, including The 48 Laws of Power, The Art of Seduction, The 33 Strategies of War, The 50th Law (with rapper 50 Cent), Mastery, The Laws of Human Nature, and The Daily Laws.

Born in 1959, Greene studied classical studies and worked a variety of jobs, before his first book was published in 1998. Greene frequently draws on analyses of past historical figures and events throughout his writing. Greene's works have been referenced by a wide variety of celebrities, political figures, and civil rights activists. He is the most banned author in prisons in the United States;[2] many prisons ban his books as a security measure.[3][4]
    """
    summary_template = """
given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)

    llm = ChatOllama(model="gemma3:270m", temperature=0)

    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})

    print(response.content)


if __name__ == "__main__":
    main()
