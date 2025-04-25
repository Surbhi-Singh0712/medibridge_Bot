
from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from connect_memory_with_llm import retriever, generate_llm_response

# Load environment variables
load_dotenv()

app = Flask(__name__)
 
# Add route to serve the HTML chatbot UI
@app.route('/')
def home():
    return render_template('chat.html')  # chat.html must be in the 'templates/' folder


#  handle the chatbot queries
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
    app.run(debug=False, host="0.0.0.0", port=port)
