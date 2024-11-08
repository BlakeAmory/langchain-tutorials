import os
from operator import itemgetter

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.messages import (
    HumanMessage,
    trim_messages,
    AIMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_openai import ChatOpenAI

LANGCHAIN_API_KEY = os.environ["LANGCHAIN_API_KEY"]
LANGCHAIN_TRACING_V2 = os.environ["LANGCHAIN_TRACING_V2"]

model = ChatOpenAI(base_url="http://localhost:1234/v1", model="gpt-3.5-turbo")

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# BEFORE the prompt template but AFTER loading previous messages from Message History.
trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm Ali"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt_template
    | model
)

with_message_history = RunnableWithMessageHistory(
    chain, get_session_history, input_messages_key="messages"
)

config = {"configurable": {"session_id": "abc2"}}

# response = with_message_history.invoke(
#     {
#         "messages": [
#             HumanMessage(content="Hi, I'm Ali"),
#         ],
#         "language": "German",
#     },
#     config=config,
# )
#
# print(response.content)

response = with_message_history.invoke(
    {"messages": messages + [HumanMessage("What's my name?")], "language": "English"},
    config=config,
)

print(response.content)


response = with_message_history.invoke(
    {
        "messages": [HumanMessage(content="what math problem did i ask?")],
        "language": "English",
    },
    config=config,
)

print(response.content)

for r in with_message_history.stream(
    {
        "messages": HumanMessage(content="hi! I'm todd. tell me a joke"),
        "language": "English",
    },
    config=config,
):
    print(r.content, end="|")

# print(store)
