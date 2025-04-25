
from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from create_memory_for_llm import load_pdf_files, create_chunks, get_embedding_model
from connect_memory_with_llm import retriever, generate_llm_response
from langchain_community.vectorstores import FAISS 


# Load environment variables
load_dotenv()

app = Flask(__name__)


# Paths
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstores/db_faiss"
HUGGINGFACE_REPO_ID = os.getenv("HUGGINGFACE_REPO_ID")

# Add route to serve the HTML chatbot UI
@app.route('/')
def home():
    return render_template('chat.html')  # chat.html must be in the 'templates/' folder

# Step 1: Automatically Process PDFs and Create Embeddings
@app.route('/process_pdfs', methods=['GET'])
def process_pdfs():
    documents = load_pdf_files(DATA_PATH)
    text_chunks = create_chunks(documents)
    embedding_model = get_embedding_model()
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)

    return jsonify({"message": "PDFs processed successfully."})

# Step 2: Query the chatbot queries
@app.route('/query', methods=['POST'])
def query():
    user_query = request.form.get('query') or request.form.get('msg')
     # Get relevant documents
    retrieved_docs = retriever.get_relevant_documents(user_query)
    context_text = "\n".join([doc.page_content for doc in retrieved_docs])

    # Create custom prompt
    CUSTOM_PROMPT_TEMPLATE = """
    Use the pieces of information provided in the context to answer user's question.
    If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
    Don't provide anything out of the given context.

    Context: {context}
    Question: {question}

    Start the answer directly. No small talk please.
    """
    final_prompt = CUSTOM_PROMPT_TEMPLATE.format(context=context_text, question=user_query)

    # Generate response
    response = generate_llm_response(final_prompt)

    return jsonify({"result": response})

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # 10000 is the default Render port
    app.run(debug=True, host="0.0.0.0", port=port)
