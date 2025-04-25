import os
from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv


# Load environment variables from .env
load_dotenv(find_dotenv())

# Step 1: Setup credentials and paths
HF_TOKEN=os.getenv("HF_TOKEN")
HUGGINGFACE_REPO_ID = os.getenv("HUGGINGFACE_REPO_ID")
DB_FAISS_PATH="vectorstores/db_faiss"


# Step 2: Initialize HuggingFace inference client
client = InferenceClient(token=HF_TOKEN)

# Step 3: Function to get LLM response using HuggingFace
def generate_llm_response(prompt: str):
    return client.text_generation(
        model=HUGGINGFACE_REPO_ID,
        prompt=prompt,
        max_new_tokens=256,
        temperature=0.5,
    )
 
 # Step 4: Define a custom prompt template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

# Step 5: Create a prompt template object (optional: use later if chaining with LangChain)
def set_custom_prompt(prompt_template:str):
    return PromptTemplate(template=prompt_template, input_variables=["context", "question"])

prompt_template = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
# Step 6: Load FAISS DB and return retriever
def get_retriever():
  embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
  db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
  return db.as_retriever(search_kwargs={'k': 3})

# Create global retriever instance to directly use in app.py
retriever = get_retriever()
