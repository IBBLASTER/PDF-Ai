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
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages.ai import AIMessage
from template import css, bot, user

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
     # if session_id not in store:
          # store[session_id] = ChatMessageHistory()
     if "chat_history" not in st.session_state:
          st.session_state.chat_history = ChatMessageHistory()
     # return store[session_id]
     return st.session_state.chat_history

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


def info_retrieval(user_query):
     embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
     new_vb = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
     retriever = new_vb.as_retriever()
     return retriever
     
def conv_rag_chain(retriever):
     llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
     system_prompt = (
          "You are an assistant for question-answering from user's document. "
          "Use the following pieces of retrieved context from the user's document to answer "
          "the question. If you don't know the answer, say that you "
          "don't know. Give a detailed answer, highlighting keypoints, but at the same "
          "time keep it concise."
          "\n\n"
          "{context}"
     )
     contextualize_q_system_prompt = (
          "Given a chat history and the latest user question "
          "which might reference context in the chat history, "
          "formulate a standalone question which can be understood "
          "without the chat history. Do NOT answer the question, "
          "just reformulate it if needed and otherwise return it as is."
     )

     contextualize_q_prompt = ChatPromptTemplate.from_messages(
         [
             ("system", contextualize_q_system_prompt),
             MessagesPlaceholder("chat_history"),
             ("human", "{input}"),
         ]
     )
     history_aware_retriever = create_history_aware_retriever(
          llm, retriever, contextualize_q_prompt
     )

     prompt = ChatPromptTemplate.from_messages(
          [
               ("system", system_prompt),
               MessagesPlaceholder("chat_history"),
               ("human", "{input}"),
          ]
     )
     qna_chain = create_stuff_documents_chain(llm, prompt)
     rag_chain = create_retrieval_chain(history_aware_retriever, qna_chain)
     conversational_rag_chain = RunnableWithMessageHistory(
          rag_chain,
          get_session_history,
          input_messages_key="input",
          history_messages_key="chat_history",
          output_messages_key="answer"
     )
     return conversational_rag_chain

def handle_qna(query, session_id):
     chain  = st.session_state.conv_chain
     if chain:
          if "chat_history" not in st.session_state:
               st.session_state.chat_history = ChatMessageHistory()
          with st.expander("Debug Information-1 !!"):
               st.write(f"Chat history before invoke: {st.session_state.chat_history.messages}")
          result = chain.invoke(
               {"input": query},
               config={"configurable": {"session_id": session_id}},
          )

          # st.session_state.chat_history.add_user_message(query)
          # st.session_state.chat_history.add_ai_message(result['answer'])
          
          with st.expander("Debug Information-2 !!"):
               st.write(f"Chat history after update: {st.session_state.chat_history.messages}")
               st.write(f"Result from chain: {result}")
          
          chat_history = st.session_state.chat_history

          for message in chat_history.messages:
            if isinstance(message, AIMessage):
                st.write(bot.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(user.replace("{{MSG}}", message.content), unsafe_allow_html=True)
     
     else:
          st.write("Conversation Chain not Initialized")

def main():
    st.set_page_config(page_title="PDF-AI", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.title("Liber AI 🤖")
    st.subheader("Upload your Document 📄 and ask questions 🙋‍♂️")
    uploaded_file = st.file_uploader("Choose a PDF File", type="pdf")

    if "conv_chain" not in st.session_state:
         st.session_state.conv_chain = None
    if "chat_history" not in st.session_state:
         st.session_state.chat_history = ChatMessageHistory()

    if uploaded_file:
        session_id = st.session_state.get("session_id", None)
        if not session_id:
             session_id  = str(uploaded_file)
             st.session_state["session_id"] = session_id

        pdf_text = get_pdf_text(uploaded_file)
        chunks = get_chunks(pdf_text)
        chunk_embeddings(chunks)
        user_question = st.text_input("Ask Your Question")
        if user_question:
            retriever = info_retrieval(user_question)
            st.session_state.conv_chain = conv_rag_chain(retriever)
            handle_qna(user_question, session_id)
        
        

if __name__ == "__main__":
     main()