from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import os
from dotenv import load_dotenv
from collections import OrderedDict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load data from text files (files are in the same directory as the script)
data_files = [
    "eco_transport.txt",
    "health_resources.txt",
    "education_support.txt",
    "matara_recycling_2025.txt"
]

documents = []
for file_path in data_files:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Split content into entries (each entry separated by blank line)
            entries = content.strip().split("\n\n")
            for entry in entries:
                if entry.strip():
                    # Parse fields
                    lines = entry.split("\n")
                    metadata = {"source": file_path}
                    doc_content = entry
                    for line in lines:
                        if line.startswith("Type:"):
                            metadata["type"] = line.replace("Type:", "").strip()
                        elif line.startswith("Name:"):
                            metadata["name"] = line.replace("Name:", "").strip()
                    documents.append(Document(page_content=doc_content, metadata=metadata))
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")

# Split the text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS index
try:
    vector_store = FAISS.from_documents(texts, embedding_model)
    logger.info("Vector Embeddings created successfully")
    vector_store.save_local("faiss_index")
except Exception as e:
    logger.error(f"Error creating vector embeddings: {e}")

# Validate the setup
try:
    vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    test_query = "Where can I find a recycling center in Matara?"
    results = vector_store.similarity_search(test_query, k=5)
    unique_results = OrderedDict()
    for doc in results:
        if doc.page_content not in unique_results:
            unique_results[doc.page_content] = doc
    final_results = list(unique_results.values())[:5]
    logger.info(f"Unique query results: {[doc.page_content for doc in final_results]}")
except Exception as e:
    logger.error(f"Error during test query: {e}")