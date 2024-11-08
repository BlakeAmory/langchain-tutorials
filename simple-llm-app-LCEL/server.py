import os

import uvicorn
from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes

LANGCHAIN_API_KEY = os.environ["LANGCHAIN_API_KEY"]
LANGCHAIN_TRACING_V2 = os.environ["LANGCHAIN_TRACING_V2"]

model = ChatOpenAI(base_url="http://localhost:1234/v1", model="gpt-4")

system_template = "Translate the following into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

parser = StrOutputParser()

chain = prompt_template | model | parser

app = FastAPI(
    title="LangChain API",
    version="1.0.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

add_routes(app=app, runnable=chain, path="/chain")


def run_app():
    uvicorn.run(app, host="localhost", port=8000)


if __name__ == "__main__":
    run_app()
