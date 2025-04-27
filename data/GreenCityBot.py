import os
import warnings
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Setup ---
warnings.filterwarnings("ignore")
load_dotenv()

# --- Check and Generate FAISS Index if Missing ---
if not os.path.exists("faiss_index"):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document

    data_files = [
        "eco_transport.txt",
        "health_resources.txt",
        "education_support.txt",
        "matara_recycling_2025.txt"
    ]

    documents = []
    for file_path in data_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            entries = content.strip().split("\n\n")
            for entry in entries:
                if entry.strip():
                    metadata = {"source": file_path}
                    lines = entry.split("\n")
                    for line in lines:
                        if line.startswith("Type:"):
                            metadata["type"] = line.replace("Type:", "").strip()
                        elif line.startswith("Name:"):
                            metadata["name"] = line.replace("Name:", "").strip()
                    documents.append(Document(page_content=entry, metadata=metadata))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(texts, embedding_model)
    vector_store.save_local("faiss_index")

# --- Embedding and Vector Store ---
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# --- Initialize the Local Model with HuggingFacePipeline ---
model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=500,
    temperature=0.7,
    device=0 if torch.cuda.is_available() else -1
)

llm = HuggingFacePipeline(pipeline=pipe)

# --- Prompt Template ---
prompt_template = """
You are GreenCity Assistant, providing accurate information on sustainable living in Matara, Sri Lanka. Use the provided context to answer the question concisely. Focus on:

1. Eco-transport, recycling, health, and education.
2. Specific details like names, locations, contacts, and descriptions.
3. If the context lacks details, suggest verifying with the source (e.g., www.matara.dist.gov.lk).
4. Decline off-topic queries politely.

Context: {context}

Question: {question}

Answer:
"""
custom_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# --- RAG Chain ---
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs={"prompt": custom_prompt},
)

# --- Response Function ---
def get_response(question):
    try:
        result = rag_chain({"query": question})
        response_text = result["result"]
        
        # Log retrieved documents for debugging
        retrieved_docs = vector_store.similarity_search(question, k=5)
        logger.info("Retrieved Documents:")
        for doc in retrieved_docs:
            logger.info(f"- {doc.page_content} (Source: {doc.metadata['source']})")
        
        # Extract answer
        answer_start = response_text.find("Answer:") + len("Answer:") if "Answer:" in response_text else 0
        answer = response_text[answer_start:].strip()
        
        # Remove prompt fragments
        unwanted_phrases = [
            "No Duplication", "No Sign-offs", "Best regards", "Streamlined", "Unique Phrasing",
            "Precision", "Topics", "Off-topic Handling", "Sustainability Focus", "Contextual Accuracy",
            "Relevance Check", "Query Context", "As a knowledgeable GreenCity Assistant"
        ]
        for phrase in unwanted_phrases:
            answer = answer.replace(phrase, "").strip()
        
        answer = " ".join(answer.split())
        return answer if answer else "I couldn't find a relevant answer. Please try rephrasing your question or check with www.matara.dist.gov.lk."
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return "An error occurred. Please try again or contact support."

# --- Streamlit App ---

# Page Styling
st.markdown(
    """
    <style>
        .appview-container .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        .stChatMessage {
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
        }
        .stChatMessage.user {
            background-color: #333333;
            color: white;
            text-align: right;
        }
        .stChatMessage.assistant {
            background-color: #4CAF50;
            color: white;
            text-align: left;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Title
st.markdown(
    """
    <h3 style='text-align: left; color: white; padding-top: 35px; border-bottom: 3px solid green;'>
        GreenCity Assistant: Sustainable Living in Matara 🌿
    </h3>
    """,
    unsafe_allow_html=True,
)

# Sidebar Content
side_bar_message = """
Hi! 👋 I'm GreenCity Assistant, here to help you live sustainably in Matara.  
Here are some areas I can assist with:
1. **Eco-Friendly Transportation** 🚌🚲
2. **Recycling Centers** ♻️
3. **Health Resources** 🏥
4. **Educational Opportunities** 📚

Feel free to ask me anything about sustainable living in Matara!
"""

with st.sidebar:
    st.title('🌱 GreenCity Assistant')
    st.markdown(side_bar_message)

# Initial Message
initial_message = """
Hi there! I'm your GreenCity Assistant 🌱  
Here are some questions you might ask me:  
🌍 What’s the most eco-friendly way to travel to Colombo?  
🌍 Where can I find a recycling center in Matara?  
🌍 Are there any free health clinics nearby?  
🌍 What free educational programs are available in Matara?  
🌍 How can I reduce my carbon footprint in the city?
"""

# Store Chat History
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]

# Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Clear Chat Button
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]
st.button('Clear Chat', on_click=clear_chat_history)

# Chat Input
if prompt := st.chat_input(placeholder="Type your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# Generate Assistant Response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Fetching sustainable recommendations for you..."):
            response = get_response(prompt)
            placeholder = st.empty()
            placeholder.markdown(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)