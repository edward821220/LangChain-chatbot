import argparse
from dotenv import load_dotenv
from styles import bold, red, blue, green
from prompt_toolkit import prompt
from prompt_toolkit.styles import Style
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import InMemoryHistory
from langchain.chains import LLMMathChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.prompts import MessagesPlaceholder, SystemMessagePromptTemplate


# import langchain
# langchain.debug = True

load_dotenv()

kb = KeyBindings()

style = Style.from_dict(
    {
        "prompt": "yellow bold",
    }
)


@kb.add("enter")
def submit(event):
    event.app.exit(result=event.app.current_buffer.text)


@kb.add("down")
def newline(event):
    buffer = event.app.current_buffer
    buffer.insert_text("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Simple command line chatbot with GPT-4"
    )

    parser.add_argument(
        "--system",
        type=str,
        help="A brief chatbot's system prompt",
        default="如果我問題用中文問的話，請一律用繁體中文(台灣)回答。",
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

    llm_math = LLMMathChain.from_llm(llm, verbose=True)

    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="Useful when you need to answer questions about current events. You should ask targeted questions.",
        ),
        Tool(
            name="Calculator",
            func=llm_math.run,
            description="useful for when you need to answer questions about math.",
        ),
    ]

    agent_kwargs = {
        "extra_prompt_messages": [
            SystemMessagePromptTemplate.from_template(args.system),
            MessagesPlaceholder(variable_name="chat_history"),
        ],
    }

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent_chain = initialize_agent(
        tools,
        llm,
        verbose=True,
        memory=memory,
        agent_kwargs=agent_kwargs,
        agent=AgentType.OPENAI_FUNCTIONS,
    )

    history = InMemoryHistory()

    while True:
        try:
            user_input = prompt(
                "You: ",
                multiline=True,
                wrap_lines=True,
                style=style,
                key_bindings=kb,
                history=history,
                enable_history_search=True,
                auto_suggest=AutoSuggestFromHistory(),
            ).strip()
            res = agent_chain.run(user_input)
            print(bold(blue("GPT: ")), bold(green(res)))
            history.append_string(user_input)

        except KeyboardInterrupt:
            print(bold(red("Exiting...")))
            break


if __name__ == "__main__":
    main()
