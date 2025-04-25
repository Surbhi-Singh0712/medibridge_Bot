import os
from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

#step 1: setup LLM(Mistral with HuggingFace)
HF_TOKEN=os.getenv("HF_TOKEN")
HUGGINGFACE_REPO_ID = os.getenv("HUGGINGFACE_REPO_ID")

# Load HF client
client = InferenceClient(token=HF_TOKEN)

# Function to send prompt to HF
def generate_llm_response(prompt: str):
    return client.text_generation(
        model=HUGGINGFACE_REPO_ID,
        prompt=prompt,
        max_new_tokens=256,
        temperature=0.5,
    )
# Step 2: Connect LLM with FAISS and Create chain

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

# Define set_custom_prompt function
def set_custom_prompt(prompt_template):
    return PromptTemplate(template=prompt_template, input_variables=["context", "question"])

#load Database
DB_FAISS_PATH="vectorstores/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Expose retriever and prompt for use in app.py
retriever = db.as_retriever(search_kwargs={'k': 3})
prompt_template = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)

#now invoke with single query
# user_query=input("Write Query Here: ")
# response=qa_chain.invoke({'query': user_query})
# print("RESULT: ", response["result"]) 

 