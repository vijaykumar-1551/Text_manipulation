import streamlit as st
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

st.title("Answer Evaluation")

# Define the user input fields for the two paragraphs
paragraph1 = st.text_input("Enter the Actual answer:", value="")
paragraph2 = st.text_input("Enter the Candidate's answer:", value="")

# Add a submit button
if st.button("Check"):
    if paragraph1 and paragraph2:
        # Compute the embeddings for the two paragraphs
        embeddings1 = model.encode(paragraph1, convert_to_tensor=True)
        embeddings2 = model.encode(paragraph2, convert_to_tensor=True)

        # Calculate the cosine similarity between the two embeddings
        cos_sim = util.pytorch_cos_sim(embeddings1, embeddings2)

        # Define a threshold for similarity
        threshold = 0.8

        # Determine whether the paragraphs are similar or not
        if cos_sim.item() > threshold:
            st.write("Correct --> similar by", cos_sim.item() * 100)
        else:
            st.write("Wrong --> similar by", 100 - (cos_sim.item() * 100))
