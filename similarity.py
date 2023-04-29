import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

st.title("Paragraph Similarity Checker")

# Define the user input fields for the two paragraphs
paragraph1 = st.text_input("Enter the first paragraph:")
paragraph2 = st.text_input("Enter the second paragraph:")

# Add a submit button
if st.button("Check Similarity"):
    if paragraph1 and paragraph2:
        # Tokenize the paragraphs and convert to PyTorch tensors
        inputs1 = tokenizer(paragraph1, padding=True, truncation=True, return_tensors="pt")
        inputs2 = tokenizer(paragraph2, padding=True, truncation=True, return_tensors="pt")

        # Get the model output for each paragraph
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

        # Calculate the cosine similarity between the two output embeddings
        cos_sim = torch.nn.functional.cosine_similarity(outputs1.last_hidden_state.mean(dim=1), outputs2.last_hidden_state.mean(dim=1))

        # Define a threshold for similarity
        threshold = 0.8

        # Determine whether the paragraphs are similar or not
        if cos_sim.item() > threshold:
            st.write("The two paragraphs are similar by:" , cos_sim.item()*100)
        else:
            st.write("The two paragraphs are not similar." , cos_sim.item()*100)
