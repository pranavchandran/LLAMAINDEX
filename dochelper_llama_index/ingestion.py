"""
Author: pranav
Date: 29-01-2024
Version: 1.0.0
"""
from dotenv import load_dotenv
import os

from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
# OpenAI
from llama_index.llms import OpenAI
# Embeddings
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index import (download_loader, ServiceContext,
                         VectorStoreIndex, StorageContext)

from llama_index.vector_stores import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

# pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# Check if the index exists
# data_abt_index: str = pc.describe_index(os.getenv("PINECONE_INDEX_NAME"))
# print(data_abt_index)

if __name__ == "__main__":
    print(os.getenv("PINECONE_INDEX_NAME"))
    UnstructuredReader = download_loader("UnstructuredReader")
    dir_reader = SimpleDirectoryReader(
        input_dir="llama_test_directory/",
        file_extractor={
            ".html": UnstructuredReader()
        }
    )
    documents = dir_reader.load_data()
    node_parser = SimpleNodeParser.from_defaults(
        chunk_size=500,
        chunk_overlap=20,
    )
    # llm
    llm = OpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )
    # embed_model
    embed_model = OpenAIEmbedding(
        model="text-embedding-ada-002",
        embed_batch_size=100
    )

    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser
    )
    # vectorstore
    vector_store = PineconeVectorStore(
        pinecone_index=pc.Index(name=os.getenv("PINECONE_INDEX_NAME")
                                )
    )
    # storage context
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    # index
    index = VectorStoreIndex.from_documents(
        documents=documents,
        service_context=service_context,
        storage_context=storage_context,
        show_progress=True
    )
    print("Finished indexing")