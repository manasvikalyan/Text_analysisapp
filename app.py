import PyPDF2 as pdf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import bert_score
from rouge_score import rouge_scorer

def extract_text(uploaded_file):
    text = ""
    if uploaded_file:
        reader = pdf.PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def calculate_similarity(text1, text2):
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0][1]

def bert_similarity(text1, text2):
    P, R, F1 = bert_score.score([text1], [text2], lang="en", verbose=True)
    return F1.item()

def rouge_similarity(text1, text2):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(text1, text2)
    return scores['rougeL'].fmeasure

def main():
    st.title("SIMILARITY CHECKER BETWEEN TWO PDF FILES")
    st.write("This app checks the similarity between two PDF files using different similarity metrics.")
    st.write("Upload two PDF files, select a similarity metric from the dropdown menu, and read about the metrics before checking the similarity.")

    uploaded_file1 = st.file_uploader("Choose a PDF file 1", type="pdf")
    uploaded_file2 = st.file_uploader("Choose a PDF file 2", type="pdf")

    st.sidebar.title("Similarity Metrics")
    st.sidebar.write("**Cosine Similarity**:")
    st.sidebar.write("Measures how similar the two documents are based on their content.")
    st.sidebar.write("**BERT Score**:")
    st.sidebar.write("Provides a similarity measure based on contextual embeddings of the documents.")
    st.sidebar.write("**ROUGE Score**:")
    st.sidebar.write("Evaluates the overlap in n-grams between the two documents.")

    similarity_metric = st.selectbox("Select Similarity Metric", ["Cosine Similarity", "BERT Score", "ROUGE Score"])

    if uploaded_file1 and uploaded_file2:
        if st.button("Check Similarity"):
            text1 = extract_text(uploaded_file1)
            text2 = extract_text(uploaded_file2)
            similarity = None
            if similarity_metric == "Cosine Similarity":
                similarity = calculate_similarity(text1, text2)
                st.write(f"The similarity between the two files is {similarity:.2f}.")
            elif similarity_metric == "BERT Score":
                bert_similarity_score = bert_similarity(text1, text2)
                st.write(f"The BERT similarity score between the two files is {bert_similarity_score:.2f}.")
            elif similarity_metric == "ROUGE Score":
                rouge_similarity_score = rouge_similarity(text1, text2)
                st.write(f"The ROUGE similarity score between the two files is {rouge_similarity_score:.2f}.")

if __name__ == "__main__":
    main()
