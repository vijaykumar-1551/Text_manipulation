import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
from transformers import pipeline 
from transformers import TFAutoModelForSeq2SeqLM
import torch

# Function to scrape Wikipedia article and summarize text
@st.cache_data(suppress_st_warning=True, allow_output_mutation=True)
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
        summarizer = pipeline("summarization", model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name, from_pt = True), tokenizer="sshleifer/distilbart-cnn-12-6")
        max_chunk_len = 1024
        chunks = [text[i:i+max_chunk_len] for i in range(0, len(text), max_chunk_len)]
        summary = ""
        for chunk in chunks:
            summary += summarizer(chunk, max_length=120, min_length=30, do_sample=False)[0]['summary_text'] + " "
        return summary

    # Save summary to file
    summary = summarizeText(article)
    with open('summary.txt', 'w') as f:
        f.write(summary)

    # Show summary in streamlit
    st.write("Summary:")
    st.write(summary)

    # Return summarized text
    return summary

# Streamlit UI
st.title("Wikipedia Article Summarizer")
url = st.text_input("Enter Wikipedia URL:")
if url:
    summary = scrapeWikiArticle(url)
    while True:
        question = st.text_input("Ask a question (type 'quit' to exit):")
        if question == "quit":
            break
        else:
            question_answerer = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")
            result = question_answerer(question=question, context=summary)
            st.write("Answer:")
            st.write(result['answer'])
