import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import datetime
import pytz

load_dotenv()
#google api key
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

#to extract texts from the uploaded documents
def get_pdf_text(pdf_docs):
    """Extract text from PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
    
#splitting extracted documents into chunks using Recursive Character textsplitter from langchain
def get_text_chunks(text):
    """Split text into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks

#Embedding text chunks for FAISS vector store and storing at faiss_index.
def get_vector_store(chunks):
    """Create a FAISS vector store from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("faiss_index")
    return vector_store
    
#setting up lagchain QA chain for user uinput and response adding prompts 
def get_conversational_chain():
    """Set up a Langchain question-answering chain."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    Context: {context}
    Question: {question}
    """
    #using gemini model and and genai client temperature at .3
    model = ChatGoogleGenerativeAI(model="gemini-pro", client=genai, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

#taking user queries and finding output using similary search 
def user_input(user_question):
    """Handle user questions and generate responses."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Check if the FAISS index exists
    if os.path.exists("faiss_index"):
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain.invoke({"input_documents": docs, "question": user_question})
    else:
        # If the FAISS index doesn't exist, display a message
        st.warning("Please upload some documents first before asking a question.")
        response = None

    return response
#clear chat message from the streamlit session
def clear_chat_history():
    """Clear the chat history stored in the Streamlit session state."""
    st.session_state.messages = [{"role": "assistant", "content": "upload some pdfs and ask me a question"}]

#this is called if the user query is call me 
def handle_call_request(user_info):
    """Handle the user's request to receive a call."""
    name = user_info.get("name")
    phone = user_info.get("phone")
    email = user_info.get("email")

    # Get current UTC time
    utc_now = datetime.datetime.utcnow()

    # Specify the user's timezone directly
    user_timezone = 'Asia/Kathmandu'  # Nepal's timezone

    local_timezone = pytz.timezone(user_timezone)
    local_now = utc_now.replace(tzinfo=pytz.utc).astimezone(local_timezone)

    # Calculate time 30 minutes ahead
    time_30_minutes_ahead = local_now + datetime.timedelta(minutes=30)

    # Format the local time and time 30 minutes ahead as required
    local_time_str = local_now.strftime("%Y-%m-%d %H:%M") + f" / {local_now.strftime('%z')} UTC"
    time_30_minutes_ahead_str = time_30_minutes_ahead.strftime("%Y-%m-%d %H:%M") + f" / {time_30_minutes_ahead.strftime('%z')} UTC"


    # Assuming st is Streamlit's module
    with st.chat_message("assistant"):
        st.write(f"Got it, thanks {name}. I'll make sure someone gives you a call at your number : {phone} at {local_time_str}. "
                f"The call will be made around {time_30_minutes_ahead_str}. I've also noted your email address: {email}.")

#collecting user infor from the form such as name phone and email with form validation only temporary storing in cache
def collect_user_info():
    """Add a form to the Streamlit UI for user information collection."""
    with st.form("user_info_form"):
        st.title("Let's connect!")
        name = st.text_input("Name")
        phone = st.text_input("Phone Number")
        email = st.text_input("Email")
        submit = st.form_submit_button("Submit")
        if submit:
            user_info = {"name": name, "phone": phone, "email": email}
            st.session_state.user_info = user_info
            st.session_state.user_info_submitted = True
            st.rerun()
            
#main funtion to handle all the frontend using streamlit
def main():
    """Set up the Streamlit user interface and handle user interactions."""
    st.set_page_config(page_title="Gemini PDF Chatbot", page_icon="🤖")

    # Check if user info has been submitted
    if "user_info_submitted" not in st.session_state:
        st.session_state.user_info_submitted = False

    # Display user info form if not submitted
    if not st.session_state.user_info_submitted:
        collect_user_info()
    else:
        # Display sidebar for PDF upload and processing
        with st.sidebar:
            st.title("Menu:")
            pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")

        # Display main chat interface
        st.title("Chat with PDF files using Gemini🤖")
        st.write("Welcome to the chat!")
        st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

        # Handle user messages and generate responses
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [{"role": "assistant", "content": "upload some pdfs and ask me a question"}]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            # Handle "who am i" and "call me" requests
            if prompt.lower() == "who am i":
                if "user_info" in st.session_state:
                    user_info = st.session_state.user_info
                    name = user_info.get("name")
                    phone = user_info.get("phone")
                    email = user_info.get("email")
                    with st.chat_message("assistant"):
                        st.write(f"According to the information you provided, your name is {name}, your phone number is {phone}, and your email is {email}.")
                else:
                    with st.chat_message("assistant"):
                        st.write("I'm afraid I don't have any information about you yet. Could you please fill out the form in the sidebar first?")

            elif "call me" in prompt.lower():
                if "user_info" in st.session_state:
                    user_info = st.session_state.user_info
                    name = user_info.get("name")
                    phone = user_info.get("phone")
                    email = user_info.get("email")
                    with st.chat_message("assistant"):
                        st.write(f"Hi {name}, confirmation will soon arrive regarding your call details?")
                    handle_call_request(user_info)
                else:
                    with st.chat_message("assistant"):
                        st.write("I'm afraid I don't have any information about you yet. Could you please fill out the form in the sidebar first?")

            # Handle "how do you work?" request
            elif "how do you work?" in prompt.lower():
                with st.chat_message("assistant"):
                    st.write("I am an AI assistant powered by Langchain and Google Generative AI (Gemini). To use me, first upload your PDF files in the sidebar and submit . Then, you can ask me questions about the content of the documents, and I will provide responses based on the information in the documents. If you ever need to contact me, you can fill out the form in the sidebar with your name, phone number, and email, and I'll make sure someone gets in touch with you.")

            # Handle other user questions
            else:
                if st.session_state.messages[-1]["role"] != "assistant":
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response = user_input(prompt)
                            placeholder = st.empty()
                            full_response = ''
                            for item in response['output_text']:
                                full_response += item
                                placeholder.markdown(full_response)
                            placeholder.markdown(full_response)
                    if response is not None:
                        message = {"role": "assistant", "content": full_response}
                        st.session_state.messages.append(message)

if __name__ == "__main__":
    main()
