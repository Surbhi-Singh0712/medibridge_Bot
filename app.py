# from flask import Flask, render_template, request, jsonify
# import os
# from create_memory_for_llm import load_pdf_files, create_chunks, get_embedding_model
# from connect_memory_with_llm import load_llm, FAISS, HuggingFaceEmbeddings, RetrievalQA, PromptTemplate

# #from file_create_memory_for_llm import load_pdf_files, create_chunks, get_embedding_model
# #from file_connect_memory_with_llm import load_llm, FAISS, HuggingFaceEmbeddings, RetrievalQA, PromptTemplate
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# app = Flask(__name__)

# # Add route to serve the HTML chatbot UI
# @app.route('/')
# def home():
#     return render_template('chat.html')  # chat.html must be in the 'templates/' folder

# # Paths
# DATA_PATH = "data/"
# DB_FAISS_PATH = "vectorstores/db_faiss"

# # Step 1: Automatically Process PDFs and Create Embeddings
#  @app.route('/query', methods=['POST'])
# def query():
#     try:
#         user_query = request.form.get('query') or request.form.get('msg')
#         if not user_query:
#             return jsonify({"error": "Query is required."}), 400
        
#         # Load the FAISS DB
#         embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        
#         # Create and invoke the QA chain
#         hf_token = os.getenv("HF_TOKEN")
#         if not hf_token:
#             return jsonify({"error": "HuggingFace token not found in environment variables."}), 400
        
#         # Pass the huggingface_repo_id here
#         llm = load_llm(HUGGINGFACE_REPO_ID)
        
#         CUSTOM_PROMPT_TEMPLATE = """
#         Use the pieces of information provided in the context to answer user's question.
#         If you dont know the answer, just say that you dont know, dont try to make up an answer. 
#         Dont provide anything out of the given context
        
#         Context: {context}
#         Question: {question}
        
#         Start the answer directly. No small talk please.
#         """
        
#         def set_custom_prompt(prompt_template):
#             return PromptTemplate(template=prompt_template, input_variables=["context", "question"])

#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type="stuff",
#             retriever=db.as_retriever(search_kwargs={'k': 3}),
#             return_source_documents=True,
#             chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
#         )

#         response = qa_chain.invoke({'query': user_query})
        
#         return jsonify({"result": response["result"]})
    
#     except Exception as e:
#         return jsonify({"error": f"Error processing query: {str(e)}"}), 500


# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template, request, jsonify
import os
from create_memory_for_llm import load_pdf_files, create_chunks, get_embedding_model
from connect_memory_with_llm import load_llm, FAISS, HuggingFaceEmbeddings, RetrievalQA, PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Add route to serve the HTML chatbot UI
@app.route('/')
def home():
    return render_template('chat.html')  # chat.html must be in the 'templates/' folder

# Paths
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstores/db_faiss"
HUGGINGFACE_REPO_ID = os.getenv("HUGGINGFACE_REPO_ID")

# Step 1: Automatically Process PDFs and Create Embeddings
@app.route('/process_pdfs', methods=['GET'])
def process_pdfs():
    documents = load_pdf_files(DATA_PATH)
    text_chunks = create_chunks(documents)

    embedding_model = get_embedding_model()
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)

    return jsonify({"message": "PDFs processed successfully."})

# Step 2: Query the Document using LLM and FAISS
@app.route('/query', methods=['POST'])
def query():
    user_query = request.form.get('query') or request.form.get('msg')
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

    llm = load_llm(HUGGINGFACE_REPO_ID)

    CUSTOM_PROMPT_TEMPLATE = """
    Use the pieces of information provided in the context to answer user's question.
    If you dont know the answer, just say that you dont know, dont try to make up an answer. 
    Dont provide anything out of the given context

    Context: {context}
    Question: {question}

    Start the answer directly. No small talk please.
    """

    def set_custom_prompt(prompt_template):
        return PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )

    response = qa_chain.invoke({'query': user_query})

    return jsonify(response["result"])

if __name__ == '__main__':
    app.run(debug=True)
