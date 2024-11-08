import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

LANGCHAIN_API_KEY = os.environ["LANGCHAIN_API_KEY"]
LANGCHAIN_TRACING_V2 = os.environ["LANGCHAIN_TRACING_V2"]
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
GOOGLE_CLOUD_PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

vectorstore = Chroma.from_documents(
    documents, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
)


# def similarity_search(query):
#     search_results = vectorstore.similarity_search(query)
#     return search_results
#
#
# def similarity_search_with_score(query):
#     search_results = vectorstore.similarity_search_with_score(query)
#     return search_results
#
#
# def similarity_search_by_vector(query):
#     embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001").embed_query(
#         query
#     )
#     search_results = vectorstore.similarity_search_by_vector(embedding)
#     return search_results
#
#
# async def asimilarity_search(query):
#     search_results = await vectorstore.asimilarity_search(query)
#     return search_results
#
#
# # Example usage
# query = "cat"
#
# # Synchronous searches
# results = similarity_search(query)
# print("similarity_search\n", results)
#
# results_with_score = similarity_search_with_score(query)
# print("similarity_search_with_score\n", results_with_score)
#
# results_by_vector = similarity_search_by_vector(query)
# print("similarity_search_by_vector\n", results_by_vector)


# # Asynchronous search (example)
# async def main():
#     search_results = await asimilarity_search(query)
#     print("asimilarity_search\n", search_results)


# asyncio.run(main())

# retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)  # select top result

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

retrieve_results = retriever.batch((["cat", "shark"]))
print("retrieve_results\n", retrieve_results)

llm = ChatGroq(model="llama3-8b-8192")

message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])

rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

response = rag_chain.invoke("tell me about cats")
print(response.content)
