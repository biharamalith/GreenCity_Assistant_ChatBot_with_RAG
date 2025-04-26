import os
import warnings
import streamlit as st
from dotenv import load_dotenv
from functools import partial
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- Setup ---
warnings.filterwarnings("ignore")
load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# --- Embedding and Vector Store ---
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# --- Safe HuggingFaceEndpoint to Remove Unsupported kwargs ---
class SafeHuggingFaceEndpoint(HuggingFaceEndpoint):
    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        kwargs.pop("stop", None)
        kwargs.pop("stop_sequences", None)
        kwargs.pop("return_full_text", None)
        kwargs.pop("watermark", None)
        return super()._call(prompt, stop=None, run_manager=run_manager, **kwargs)

llm = SafeHuggingFaceEndpoint(
    repo_id="google/flan-t5-large",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=250,
)

# --- Prompt Template ---
prompt_template = """
As a knowledgeable GreenCity Assistant, your role is to provide accurate, real-time recommendations to urban residents in Matara, Sri Lanka, to enhance sustainability and well-being. Follow these directives to ensure optimal user interactions:
1. Precision: Respond only with directly relevant info from the provided database.
2. Topics: Focus only on eco-transport, recycling, health, and education resources.
3. Off-topic Handling: Politely decline off-topic queries.
4. Sustainability Focus: Promote eco-friendly practices.
5. Contextual Accuracy: Stick strictly to query context.
6. Relevance Check: Guide user to refine irrelevant queries.
7. No Duplication: Keep each response unique.
8. Streamlined: No unnecessary comments or closings.
9. No Sign-offs: Avoid phrases like "Best regards."
10. Unique Phrasing: No repeated sentence structures.

Query Context:
{context}

Question: {question}

Answer:
"""

custom_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# --- RAG Chain ---
rag_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": custom_prompt},
)

# --- Response Function ---
def get_response(question):
    result = rag_chain({"query": question})
    response_text = result["result"]
    answer_start = response_text.find("Answer:") + len("Answer:")
    answer = response_text[answer_start:].strip()
    return answer

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
