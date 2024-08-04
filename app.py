from dotenv import load_dotenv
import os
import google.generativeai as genai
from PyPDF2 import PdfReader
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


def get_pdf_text(uploaded_file):
        reader = PdfReader(uploaded_file)
        pdf_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pdf_text += page_text
        return pdf_text


def get_chunks(pdf_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_text(pdf_text)
    return chunks

def chunk_embeddings(chunks):
     embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
     vector_base = FAISS.from_texts(chunks, embedding=embeddings)
     vector_base.save_local("faiss_index")


def info_retrival(user_query):
     embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
     new_vb = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
     docs = new_vb.similarity_search(user_query)
     return docs
     
def gemini_response(context, query):
     llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
     chain = load_qa_chain(llm, chain_type="stuff")
     result = chain.invoke({"input_documents": context, "question": query})
     yield result["output_text"]

def main():
    st.set_page_config(page_title="PDF-AI")
    st.title("Liber AI 🤖")
    st.subheader("Upload your Document 📄 and ask questions 🙋‍♂️")
    uploaded_file = st.file_uploader("Choose a PDF File", type="pdf")
    if uploaded_file:
        pdf_text = get_pdf_text(uploaded_file)
        chunks = get_chunks(pdf_text)
        chunk_embeddings(chunks)
        user_question = st.text_input("Ask Your Question")
        if user_question:
            relevant_chunks = info_retrival(user_question)
            response = gemini_response(relevant_chunks, user_question)
            st.write(response)
        


if __name__ == "__main__":
     main()