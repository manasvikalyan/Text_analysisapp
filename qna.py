import streamlit as st
import pdfplumber
import spacy
from collections import Counter

# Load English language model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(uploaded_file):
    text = ""
    if uploaded_file:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
    return text

def generate_questions(text):
    # Process the text using SpaCy
    doc = nlp(text)
    
    # Extract nouns and proper nouns as potential entities
    entities = [ent.text for ent in doc.ents if ent.label_ in ["NORP", "PERSON", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"]]
    entity_counts = Counter(entities)
    
    # Generate questions from entities
    questions = []
    for entity, count in entity_counts.most_common(5):  # Adjust the number of questions as needed
        questions.append(f"What is {entity}?")
        questions.append(f"Who is {entity}?")
        questions.append(f"When did {entity} occur?")
        questions.append(f"Where is {entity} located?")
        questions.append(f"What are the features of {entity}?")
    
    return questions

def main():
    st.title("PDF Question Answering System")
    st.write("Upload a PDF file and generate questions based on its contents.")

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file)
        st.write(f"Extracted Text:\n{text}")

        if st.button("Generate Questions"):
            questions = generate_questions(text)
            st.write("Generated Questions:")
            for idx, question in enumerate(questions, start=1):
                st.write(f"{idx}. {question}")

if __name__ == "__main__":
    main()
