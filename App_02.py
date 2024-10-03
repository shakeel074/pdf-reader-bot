import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

# Sidebar contents
with st.sidebar:
    st.title("LLM Chat App")
    st.markdown('''
    This App is an LLM-Powered chatbot built using:
    - [Streamlit for the frontend](https://streamlit.io/)
    - [Langchain](https://streamlit.io/)
    - [OpenAI](https://streamlit.io/)
    ''')
    add_vertical_space(5)
    st.write('Made with by [prompt Engineer]')

def main():
    st.header("Chat with PDF")
    
    load_dotenv()
    
    # Initialize vectorstore
    vectorstore = None
    
    # Upload a PDF File
    pdf = st.file_uploader('Upload your PDF', type='pdf')
    
    if pdf is not None:
        st.write(f"Uploaded file: {pdf.name}")  # Check if pdf is not None before accessing name
        
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )    
        chunks = text_splitter.split_text(text=text)
        
        # Embeddings
        embeddings = OpenAIEmbeddings()
        store_name = pdf.name[:-4]
        
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vectorstore = pickle.load(f)
            st.write("Embeddings loaded from the disk")
        else:
            vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vectorstore, f)
            st.write("Embeddings computations completed")
    else:
        st.write("Please upload a PDF file.")
        
    # Accept user question/query
    query = st.text_input("Ask a question about the PDF")
    #  and vectorstore is not None
    if query:
        docs = vectorstore.similarity_search(query=query, k=3)
        
        llm =OpenAI()
        chain = load_qa_chain(llm =llm,chain_type = "stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents = docs,question =query)
            print(cb)
        st.write(response)
       
if __name__ == '__main__':
    main()
