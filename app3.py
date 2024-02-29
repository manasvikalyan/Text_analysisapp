import PyPDF2 as pdf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import bert_score
from rouge_score import rouge_scorer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from difflib import SequenceMatcher
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

st.set_page_config(page_title="Streamlit Sentiment App", page_icon="static/res/favicon.png")


# Initialize the model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

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

def highlight_similarity(text1, text2):
    matcher = SequenceMatcher(None, text1, text2)
    matches = matcher.get_matching_blocks()

    highlighted_text = ""
    for match in matches:
        start1 = match.a
        end1 = match.a + match.size
        start2 = match.b
        end2 = match.b + match.size
        # Highlight the matching subsequence
        highlighted_text += text1[start1:end1] + '\n'
        highlighted_text += text2[start2:end2] + '\n\n'
    
    return highlighted_text


def generate_summary(text):
    # Encode the text
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1000, truncation=True)
    
    # Generate the summary
    outputs = model.generate(inputs, max_length=1000, min_length=100, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    # Decode the summary
    summary = tokenizer.decode(outputs[0])
    
    return summary


def predict_sentiment(text, threshold_positive, threshold_negative):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)

    threshold_positive = float(threshold_positive)
    threshold_negative = float(threshold_negative)

    if sentiment_scores.get("compound", 0) >= threshold_positive:
        return "Positive"
    elif sentiment_scores.get("compound", 0) <= threshold_negative:
        return "Negative"
    else:
        return "Neutral"


def main():
    st.title("Text Analysis App")
    st.write("This app checks the similarity between two PDF files using different similarity metrics or generates a summary for a single document or does the sentiment analyis.")
    st.write("Upload PDF files, select an option from the dropdown menu, and proceed accordingly.")
    
    
    option = st.selectbox("Select Option", ["Check Similarity", "Generate Summary", "Sentiment Analysis"])

    if option == "Check Similarity":
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

                st.write("Highlighted Similarity:")
                st.write(highlight_similarity(text1, text2))

    elif option == "Generate Summary":
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file:
            if st.button("Generate Summary"):
                text = extract_text(uploaded_file)
                summary = generate_summary(text)
                st.write("Summary:")
                st.write(summary)
    elif option == "Sentiment Analysis":
        threshold_positive = st.number_input("Threshold for Positive Sentiment:", value=0.05, step=0.01)
        threshold_negative = st.number_input("Threshold for Negative Sentiment:", value=-0.05, step=0.01)
        uploaded_file = st.file_uploader("Upload PDF Document")

        if uploaded_file:
            pdf_reader = pdf.PdfReader(uploaded_file)
            positive_count = 0
            negative_count = 0
            neutral_count = 0

            for page in pdf_reader.pages:
                text = page.extract_text()
                sentences = text.split(".")
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence:
                        sentiment = predict_sentiment(sentence, threshold_positive, threshold_negative)
                        if sentiment == "Positive":
                            positive_count += 1
                        elif sentiment == "Negative":
                            negative_count += 1
                        else:
                            neutral_count += 1

            st.write("Positive Sentences:", positive_count)
            st.write("Negative Sentences:", negative_count)
            st.write("Neutral Sentences:", neutral_count)

            labels = ["Positive", "Negative", "Neutral"]
            sizes = [positive_count, negative_count, neutral_count]

            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
            ax.axis("equal")
            ax.set_title("Sentiment Distribution")

            st.pyplot(fig)
            
if __name__ == "__main__":
    main()
