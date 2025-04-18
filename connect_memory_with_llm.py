import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

#step 1: setup LLM(Mistral with HuggingFace)
HF_TOKEN=os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":512}
    )
    return llm

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
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

#create QA chain
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

#now invoke with single query
user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"]) 


# import os 
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv())

# # Step 1: Setup LLM (Mistral with HuggingFace)
# HF_TOKEN = os.environ.get("HF_TOKEN")
# HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# def load_llm(huggingface_repo_id):
#     """Load the LLM from HuggingFace."""
#     llm = HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         task="text-generation",
#         temperature=0.5,
#         model_kwargs={"token": HF_TOKEN, "max_length": 512}
#     )
#     return llm

# # Step 2: Define Prompt Template for Custom Prompt
# CUSTOM_PROMPT_TEMPLATE = """
# Use the pieces of information provided in the context to answer user's question.
# If you dont know the answer, just say that you dont know, dont try to make up an answer. 
# Dont provide anything out of the given context

# Context: {context}
# Question: {question}

# Start the answer directly. No small talk please.
# """

# def set_custom_prompt(prompt_template):
#     """Set the custom prompt template for the LLM."""
#     return PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# # Step 3: Load the FAISS Database
# DB_FAISS_PATH = "vectorstores/db_faiss"
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# def load_faiss_db():
#     """Load the FAISS database."""
#     db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
#     return db

# # Step 4: Create QA Chain
# def create_qa_chain(db):
#     """Create a QA Chain for interacting with the LLM."""
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=load_llm(HUGGINGFACE_REPO_ID),
#         chain_type="stuff",
#         retriever=db.as_retriever(search_kwargs={'k': 3}),
#         return_source_documents=True,
#         chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
#     )
#     return qa_chain

# # Now, create a function to invoke a query
# def query_qa_chain(user_query):
#     """Query the QA chain with a user input."""
#     db = load_faiss_db()  # Load the FAISS DB
#     qa_chain = create_qa_chain(db)  # Create the QA chain
#     response = qa_chain.invoke({'query': user_query})
#     return response["result"]

# # If this file is executed directly, you can run a test query (optional)
# if __name__ == "__main__":
#     user_query = input("Write Query Here: ")
#     response = query_qa_chain(user_query)
#     print("RESULT: ", response)
