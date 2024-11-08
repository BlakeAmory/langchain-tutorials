# LangChain Tutorials

This repository contains a collection of tutorials demonstrating the use of LangChain with various APIs and models. These examples are designed to help you understand how to integrate LangChain with free API keys such as `GOOGLE_API_KEY`, `GROQ_API_KEY`, and Ollama models.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Tutorials](#tutorials)
  - [Agent](#agent)
  - [Conversational RAG](#conversational-rag)
  - [Graph RAG](#graph-rag)
  - [Local RAG](#local-rag)
  - [PDF-QA](#pdf-qa)
  - [Query Analysis](#query-analysis)
  - [SQL Agent](#sql-agent)
  - [Simple LLM App](#simple-llm-app)
  - [Vector Stores & Retrievers](#vector-stores-retrievers)
- [Contributing](#contributing)
- [License](#license)

## Overview

LangChain is a framework designed to facilitate the development of applications powered by large language models (LLMs). This repository showcases practical examples and implementations of LangChain across different use cases, utilizing freely available API keys.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/BlakeAmory/langchain-tutorials.git
    cd langchain-tutorials
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up your environment variables by copying the example file:
    ```bash
    cp .env.example .env
    ```
    Fill in the `.env` file with your API keys.

## Tutorials

Each folder in this repository contains a tutorial focused on a specific use case. Below is a brief description of each tutorial:

- **Agent**: Implementation of a basic agent that interacts with users.
- **Conversational RAG**: Demonstrates the use of Retrieval-Augmented Generation (RAG) for conversational tasks.
- **Graph RAG**: Explores the integration of RAG with graph-based data structures.
- **Local RAG**: Shows how to use RAG with locally stored data.
- **PDF-QA**: Provides an example of question answering (QA) over PDF documents.
- **Query Analysis**: Analyzes user queries to determine intent and context.
- **SQL Agent**: Integrates SQL queries with LangChain for database interactions.
- **Simple LLM App**: A simple application using a large language model.
- **Vector Stores & Retrievers**: Demonstrates the use of vector stores for efficient information retrieval.

## Contributing

Contributions are welcome! If you have any improvements or additional examples you'd like to share, please fork the repository, create a new branch, and submit a pull request.

## License

This project is under MIT license:

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
