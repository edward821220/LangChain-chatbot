import argparse
from dotenv import load_dotenv
from langchain.chains import LLMMathChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.prompts import MessagesPlaceholder, SystemMessagePromptTemplate

load_dotenv()


def bold(text):
    bold_start = "\033[1m"
    bold_end = "\033[0m"
    return bold_start + text + bold_end


def green(text):
    green_start = "\033[32m"
    green_end = "\033[0m"
    return green_start + text + green_end


def blue(text):
    blue_start = "\033[34m"
    blue_end = "\033[0m"
    return blue_start + text + blue_end


def red(text):
    red_start = "\033[31m"
    red_end = "\033[0m"
    return red_start + text + red_end


def main():
    parser = argparse.ArgumentParser(
        description="Simple command line chatbot with GPT-4"
    )

    parser.add_argument(
        "--system",
        type=str,
        help="A brief chatbot's system prompt",
        default=None,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="GPT Model to use",
        default="gpt-3.5-turbo",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Model temperature",
        default=0,
    )

    args = parser.parse_args()

    search = GoogleSerperAPIWrapper()

    llm = ChatOpenAI(model=args.model, temperature=args.temperature)

    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="Useful when you need to answer questions about current events. You should ask targeted questions.",
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math",
        ),
    ]

    agent_kwargs = {
        "extra_prompt_messages": [
            MessagesPlaceholder(variable_name="chat_history"),
        ],
    }
    if args.system:
        agent_kwargs["extra_prompt_messages"].insert(
            0, SystemMessagePromptTemplate.from_template(args.system)
        )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent_chain = initialize_agent(
        tools,
        llm,
        verbose=False,
        memory=memory,
        agent_kwargs=agent_kwargs,
        agent=AgentType.OPENAI_FUNCTIONS,
    )

    while True:
        try:
            user_input = input(bold(blue("You: ")))
            res = agent_chain.run(user_input)
            print(bold(red("GPT: ")), green(res))

        except KeyboardInterrupt:
            print("Exiting...")
            break


if __name__ == "__main__":
    main()
