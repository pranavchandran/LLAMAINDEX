"""
Author: pranav
Date: 03-02-2024
Version: 1.0.0
"""
from typing import List
from dotenv import load_dotenv
import os
from llama_index import download_loader, VectorStoreIndex, ServiceContext
from llama_index.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.vector_stores import PineconeVectorStore
from pinecone import Pinecone
load_dotenv()

# pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index(name=os.getenv("PINECONE_INDEX_NAME"))

# vectorstore
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# Check if the index exists
# data_abt_index: str = pc.describe_index(os.getenv("PINECONE_INDEX_NAME"))
# print(data_abt_index)

# Load the documents
# index = VectorStoreIndex.from_vector_store(
#     vector_store=vector_store
# )

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager(handlers=[llama_debug])
service_context = ServiceContext.from_defaults(
    callback_manager=callback_manager
)

# index
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    service_context=service_context
)

# Query the index
query = "What is “Agent-like” Components within LlamaIndex?"

query_engine = index.as_query_engine()

response = query_engine.query(query)

print(response)


if __name__ == '__main__':
    print("RAG")
