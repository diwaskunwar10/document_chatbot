This Streamlit application allows you to interactively ask questions based on the content of uploaded PDF files using Gemini, a conversational AI model. Gemini uses Google's Generative AI to generate responses.
How to Use:

    Upload PDF files using the file uploader.
    Ask your question in the text input field.
    Click on "Submit & Process" to process the PDF files and receive a response to your question.

Installation:

To run this application locally, make sure you have the following dependencies installed:

pip install -r requirements.txt

Requirements:

    Streamlit
    google-generativeai
    python-dotenv
    langchain
    PyPDF2
    FAISS
    langchain_google_genai

Usage:


streamlit run chat.py

Note:

Ensure you have set up your Google API key in a .env file to utilize Google's Generative AI.
