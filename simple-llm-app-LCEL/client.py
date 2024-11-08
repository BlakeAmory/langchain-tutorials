from langserve import RemoteRunnable


remote_chain = RemoteRunnable(url="http://localhost:8000/chain/")

print(
    remote_chain.invoke(
        {
            "language": "German",
            "text": "LangChain is a Python library that allows you to chain together different natural language "
            "processing (NLP) models to create more complex NLP tasks.",
        }
    )
)
