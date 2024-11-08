# Import relevant functionality
import asyncio
import os

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

LANGCHAIN_API_KEY = os.environ["LANGCHAIN_API_KEY"]
LANGCHAIN_TRACING_V2 = os.environ["LANGCHAIN_TRACING_V2"]
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
GOOGLE_CLOUD_PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]

# Create the agent
memory = MemorySaver()
model = ChatGroq(model="llama3-8b-8192")
search = TavilySearchResults(max_results=2)
tools = [search]
agent_executor = create_react_agent(model, tools, checkpointer=memory).with_config(
    {"run_name": "Agent"}
)

# Use the agent
config = {"configurable": {"thread_id": "abc123"}}
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im bob! and i live in sf")]}, config
):
    print(chunk)
    print("----")

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats the weather where I live?")]}, config
):
    print(chunk)
    print("----")


# async def main():  # New async function
#     async for event in agent_executor.astream_events(
#         {"messages": [HumanMessage(content="whats the weather in sf?")]}, version="v1", config=config
#     ):
#         kind = event["event"]
#         if kind == "on_chain_start":
#             if (
#                 event["name"] == "Agent"
#             ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
#                 print(
#                     f"Starting agent: {event['name']} with input: {event['data'].get('input')}"
#                 )
#         elif kind == "on_chain_end":
#             if (
#                 event["name"] == "Agent"
#             ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
#                 print()
#                 print("--")
#                 print(
#                     f"Done agent: {event['name']} with output: {event['data'].get('output')['output']}"
#                 )
#         if kind == "on_chat_model_stream":
#             content = event["data"]["chunk"].content
#             if content:
#                 # Empty content in the context of OpenAI means
#                 # that the model is asking for a tool to be invoked.
#                 # So we only print non-empty content
#                 print(content, end="|")
#         elif kind == "on_tool_start":
#             print("--")
#             print(
#                 f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
#             )
#         elif kind == "on_tool_end":
#             print(f"Done tool: {event['name']}")
#             print(f"Tool output was: {event['data'].get('output')}")
#             print("--")


# if __name__ == "__main__":
#     asyncio.run(main())  # Run the async function
