import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

VECTOR_DB_PATH = "vector_db"
KNOWLEDGE_BASE_PATH = "data/IT_Act_2000.pdf"

def setup_retriever():
    """Create or load ChromaDB retriever for IT Act 2000."""
    # Load IT Act
    loader = PyMuPDFLoader(KNOWLEDGE_BASE_PATH)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Vectorstore (persistent)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )

    return vectordb.as_retriever()
