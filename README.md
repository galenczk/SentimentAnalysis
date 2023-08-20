# SentimentAnalysis

## Project Overview

I developed this Sentiment Analysis Web Application to learn about machine learning (ML) and the current ecosystem surrounding machine learning and natural language processing. 

This project facilitates sentiment analysis on user-provided text inputs using two distinct routes: a pretrained BERT model (110M trainable parameters) facilitated through a HuggingFace Transformer Pipeline, and a model that I created and trained in PyTorch (22M trainable parameters). 
Investigating and implementing each of these two methods of applying ML technology has provided me a great introduction to these techonologies. Through this project I have gained a good understanding of the ease affored by HuggingFace Pipelines, as well as the complexity of the machine
learning paradigm and its effective application.

    Programming Language: Python and JavaScript
    
    Frameworks and Libraries: 
    - PyTorch, 
    - Numpy, 
    - HuggingFace Transformers, 
    - Flask, 
    - Preact, 
    - TailwindCSS
    - Formik, 
    - Axios

![image](https://github.com/galenczk/SentimentAnalysis/assets/73518586/57f2d7e4-f12d-41a8-a60d-4b29830a7c27)

![image](https://github.com/galenczk/SentimentAnalysis/assets/73518586/2489edca-86d9-4969-9e1c-59196d68ee63)


## Installation and Setup

To set up the project locally, follow these steps:

    Clone the GitHub repository: https://github.com/galenczk/CS469_sentimentAnalysis
    Open two terminal processes.
    In the first terminal, navigate to the "backend" directory: cd backend
    Install required dependencies: pip install -r requirements.txt
    Run the Flask server: py server.py
    Open the provided IP address (e.g., http://127.0.0.1:5000) in a browser to confirm the server is running.
    In the second terminal, navigate to the "frontend" directory: cd frontend
    Install frontend dependencies: npm install
    Start the development server: npm run dev
    Open the provided IP address (e.g., http://127.0.0.1:5173) in a browser to access the application.

## Usage

    Enter a text input into the field in the center of the page.
    Choose either the BERT model or "my model" for sentiment analysis.
    Click the "Analyze" button to obtain the sentiment analysis results.
    To clear the output, click the "Clear Output" button.

## Credits

    This project was developed independently by myself, Galen Ciszek.
