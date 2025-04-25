
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS




# Path to PDF folder and FAISS DB
DATA_PATH="data/"
DB_FAISS_PATH = "vectorstores/db_faiss"
#step 1: Load raw PDF(s)
def load_pdf_files(data):
    loader=DirectoryLoader(data,
                           glob='*.pdf',
                           loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

#documents=load_pdf_files(data=DATA_PATH)
#print("Length of PDF pages: ",len(documents))


#step 2:Create Chunks

def create_chunks(documents):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=50)
    return text_splitter.split_documents(documents)
    

# text_chunks=create_chunks(extracted_data=documents)
#print("length of Text Chunks: ",len(text_chunks))

#step 3:Create Vector Embeddings
def get_embedding_model():
      return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
   

if __name__ == "__main__":
    docs = load_pdf_files(DATA_PATH)
    chunks = create_chunks(docs)
    model = get_embedding_model()
    db = FAISS.from_documents(chunks, model)
    db.save_local(DB_FAISS_PATH)
    print("FAISS DB created and saved.")
#step 4:store ewmbeddings in FAISS

# DB_FAISS_PATH="vectorstores/db_faiss"
#  db = FAISS.from_documents(text_chunks, embedding_model)
# db.save_local(DB_FAISS_PATH)
