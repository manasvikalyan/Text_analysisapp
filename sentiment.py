import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
import PyPDF2
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Streamlit Sentiment App", page_icon="static/res/favicon.png")

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
    st.title("Sentiment Analysis App")

    threshold_positive = st.number_input("Threshold for Positive Sentiment:", value=0.05, step=0.01)
    threshold_negative = st.number_input("Threshold for Negative Sentiment:", value=-0.05, step=0.01)

    uploaded_file = st.file_uploader("Upload PDF Document")

    if uploaded_file:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
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
