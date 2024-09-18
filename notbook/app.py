import os
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
    load_index_from_storage,
    Settings,
)
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.callbacks import CallbackManager
import google.generativeai as genai
import chainlit as cl

# Set Google API Key
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize storage context and index
try:
    # Rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    # Load index
    index = load_index_from_storage(storage_context)
except:
    # Create new index if storage context not found
    documents = SimpleDirectoryReader("./data").load_data(show_progress=True)
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist()

# Chat start event handler
@cl.on_chat_start
async def start():
    # Configure LLM and embedding models with Gemini
    Settings.llm = Gemini(
        model="gemini-pro", temperature=0.1, max_tokens=1024, streaming=True
    )
    Settings.embed_model = GeminiEmbedding(model="models/embedding-001")
    Settings.context_window = 4096

    # Create service context with callback manager
    service_context = ServiceContext.from_defaults(
        callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()])
    )

    # Set up the query engine
    query_engine = index.as_query_engine(
        streaming=True, similarity_top_k=2, service_context=service_context
    )

    # Store the query engine in the user session
    cl.user_session.set("query_engine", query_engine)

    # Send initial greeting message
    await cl.Message(
        author="Assistant", content="Hello! I'm an AI assistant. How may I help you?"
    ).send()

# Handle incoming user messages
@cl.on_message
async def main(message: cl.Message):
    # Retrieve query engine from user session
    query_engine = cl.user_session.get("query_engine")  # type: RetrieverQueryEngine

    # Create an empty message object for the assistant's response
    msg = cl.Message(content="", author="Assistant")

    # Perform the query asynchronously
    res = await cl.make_async(query_engine.query)(message.content)

    # Stream response tokens as they are generated
    for token in res.response_gen:
        await msg.stream_token(token)

    # Send the final message
    await msg.send()
