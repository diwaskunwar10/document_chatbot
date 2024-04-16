This Streamlit application allows you to interactively ask questions based on the content of uploaded PDF files using Gemini, a conversational AI model. Gemini uses Google's Generative AI to generate responses.
How to Use:

Upload PDF files using the file uploader.
Ask your question in the text input field.
Click on "Submit & Process" to process the PDF files and receive a response to your question.

Installation:

pull this repo 
create and goto folder and initialize git		

	git init

then pull this repo

 	git pull https://github.com/diwaskunwar10/document_chatbot.git
  
To run this application locally, you need to install the required dependence in requirements.txt:

first create a virtual environment

	python -m venv venv

activate the environment 

	source venv/bin/activate

install the requirements 

	pip install -r requirements.txt

Setting Up Google API Key:
you can create tour api key at makersuite google  https://aistudio.google.com/app/apikey
after creating your api key  create a .env file
  
  	touch .env
  
in the .env file add the api key
					
	GOOGLE_API_KEY="your_api_key_here"(Replace your_api_key_here with your actual Google API key)  

Usage:

you can simply run this application using

	streamlit run chat.py

Note:

Ensure you have set up your Google API key in a .env file to utilize Google's Generative AI.
after starting streamlit you need to fill the form and sumbit to actually go to the chat dashboard 

