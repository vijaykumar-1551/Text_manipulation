import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
from transformers import pipeline 
from transformers import AutoModelForSeq2SeqLM
import torch

# Function to scrape Wikipedia article and summarize text
@st.cache(suppress_st_warning=True)
def scrapeWikiArticle(url):
    response = requests.get(url=url)
    soup = BeautifulSoup(response.content, 'html.parser')

    title = soup.find(id="firstHeading")
    st.write(f"Title: {title.text}")

    allLinks = soup.find(id="bodyContent").find_all("p")
    article = ""
    for link in allLinks:
        article += link.text

    # Remove invalid characters
    article = re.sub(r'[^\x00-\x7F]+', ' ', article)

    # Summarize text
    def summarizeText(text):
        model_name = "sshleifer/distilbart-cnn-12-6"
        summarizer = pipeline("summarization", model = AutoModelForSeq2SeqLM.from_pretrained(model_name), tokenizer="sshleifer/distilbart-cnn-12-6")
        max_chunk_len = 512
        chunks = [text[i:i+max_chunk_len] for i in range(0, len(text), max_chunk_len)]
        summary = ""
        for chunk in chunks:
            summary += summarizer(chunk, max_length=120, min_length=30, do_sample=False)[0]['summary_text'] + " "
        return summary


    # Show summary in streamlit
    summary = summarizeText(article)
    st.write("Summary:")
    st.write(summary)

    # Return summarized text
    return summary

# Streamlit UI
st.title("Wikipedia Article Summarizer")
url_input = st.text_input("Enter Wikipedia URL:", key="url_input")
if url_input:
    summary = scrapeWikiArticle(url_input)
    while True:
        question_input = st.text_input("Ask a question (type 'quit' to exit):", key="question_input")
        if question_input == "quit":
            break
        else:
            question_answerer = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")
            result = question_answerer(question=question_input, context=summary)
            st.write("Answer:")
            st.write(result['answer'])

