"""
Author: pranav
Date: 27-01-2024
Version: 1.0.0
"""
from typing import List
import os

from llama_index import download_loader, VectorStoreIndex
from llama_index.readers.web import SimpleWebPageReader


def main(url: str) -> None:
    """
    :param url:
    :return:
    """
    documents = SimpleWebPageReader(html_to_text=True).load_data(urls=[url])
    index = VectorStoreIndex.from_documents(documents=documents)
    query_engine = index.as_query_engine()
    response = query_engine.query("What is LLama Index?")
    print(response)

if __name__ == '__main__':
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    main(url='https://medium.com/poatek/building-open-source-llm-based-chatbots-using-llama-index-e6de9999ee76')