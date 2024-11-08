import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def process_pdf(file):
    # Create a temporary file to store the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        file.save(temp_file.name)  # Save the uploaded file to the temporary file
        temp_file.close()  # Close the file to release the file handle

    # Load the PDF document using PyPDFLoader
    loader = PyPDFLoader(file_path=temp_file.name)
    documents = loader.load()  # Extract the documents from the PDF

    # Split the document into chunks for processing
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents)

    # Create embeddings for the document chunks using Google's Generative AI model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GOOGLE_API_KEY")  # API key from environment variables
    )

    # Create a FAISS vector store to store the embeddings
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index")  # Save the vector store locally

    # Load the vector store from the saved file for future use
    new_vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Delete the temporary file to clean up
    os.remove(temp_file.name)

    return new_vectorstore  # Return the vector store for future retrieval
