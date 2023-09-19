import streamlit as st
import spacy
import en_core_web_sm
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

st.header("Trip Advisor reviews - Cyprus :flag-cy:")
st.subheader("**Sentiment analysis**")
st.divider()

st.write("The project aims to explore:")
st.markdown("- text statistics like most frequent parts of speech (POS), words (tokens), reviews' length")
st.markdown("- rating distribution as the background for text analysis")
st.markdown("- sentiment analysis of the reviews")
st.markdown("- reviews' mood classifcation.")

"---"

st.text("The obtained dataset looks like this...")


@st.cache_data

def load_data():
    file = "file.xlsx"
    df = pd.read_excel(file, sheet_name="Kamara")
    return df

# Load the data using the cached function
data = load_data()

# Display the first 5 rows of the DataFrame
st.write(data.head(5))
"---"

st.write("Select the review id 0-500 to print it :")

#function to take the user input and print the review with taken input index
row_index = st.number_input("Enter a row index:",
                            min_value=0, max_value=len(data)-1,
                            value=0,
                            step=1)

if st.button("Print"):
    if row_index >= 0 and row_index < len(data):
        review = data.at[row_index, "review"]
        st.write(review)

    else:
        st.error("The dataset does not have such index. Please indicate index within 0-500")

st.divider()
st.subheader("**Rating distribution**")

fig = px.histogram(data, x = "rating",
                   color = "rating",
                   title=f"Rating marks countplot")

st.plotly_chart(fig)

values = data.rating.value_counts(normalize = True).round(2)
labels = ["5", "4", "3", "2", "1"]

fig = go.Figure(data = [go.Pie(labels = labels,
                               values = values)])

st.plotly_chart(fig)

st.markdown("*It can be seen that almost the half of the reviews has the highest rating, which is 5.*")
st.markdown("*That gives a clue that the majority of the reviews :red[must] turn out to be :green[positive] as well.*")
st.markdown("*The dataset can be explored in terms of the consistency between sentiment analysis output and the rating mark.*")

"---"
st.subheader("**Text exploration**")

st.text("Most of the reviews do not exceed the 50 words per review.")

#creating the words count column
data["words_count"] = data.review.apply(lambda n: len(n.split()))

word_counts = data.words_count
fig = ff.create_distplot([word_counts], group_labels=["Word Counts"])

st.plotly_chart(fig)

#text preprocessing

nlp = spacy.load("en_core_web_sm")

selected_pos = st.selectbox("Select parts of speech:",
                            ["ADJ", "NOUN", "VERB"],
                            index = 0)

@st.cache(suppress_st_warning=True)
def preprocess_review(review, pos_filter):
    doc = nlp(review)
    preprocessed_words = [token.lemma_ for token in doc if not token.is_stop
                          and not token.is_punct
                          and token.pos_ in pos_filter]
    preprocessed_text = " ".join(preprocessed_words)
    return preprocessed_text

preprocessed_reviews = data["review"].apply(lambda x: preprocess_review(x, [selected_pos]))


data["preprocessed_reviews"] = preprocessed_reviews

text = " ".join(data["preprocessed_reviews"])

words = text.split()
word_freq = Counter(words)

word_freq_df = pd.DataFrame(word_freq.items(), columns=["Word", "Frequency"])
word_freq_df = word_freq_df.sort_values(by="Frequency", ascending= False)

fig = px.bar(
    word_freq_df.head(10),  # Display the top 10 most frequent words
    x="Frequency",
    y="Word",
    orientation="h",
    title= f"Top 10 Most Frequent {selected_pos.upper()}S",
    labels={"Frequency": "Word Frequency", "Word": "Word"},
)

st.plotly_chart(fig)

# topic modeling
"---"

st.subheader("**Topics discovered in text corpus**")

import transformers
from transformers import pipeline
import bertopic
from bertopic import BERTopic

topic_num = int(st.number_input("Specify the number of topics in range 1-10",
                                min_value=1,
                                max_value=10,
                                step=1))

topic_model = BERTopic(nr_topics = topic_num)

@st.cache(suppress_st_warning=True)
def transform_topics(data):
    topics, probs = topic_model.fit_transform(data.preprocessed_reviews)
    return topics, probs

topic_model.transform_topics(data.preprocessed_reviews)
fig = topic_model.visualize_barchart()
st.plotly_chart(fig)

"---"
st.subheader("**Sentiment analysis & emotion detection**")

@st.cache_data(suppress_st_warning = True)
def analyze_sentiment_bert(review):
    classifier = pipeline('sentiment-analysis')
    result = classifier(review)
    return result[0]

@st.cache_data(suppress_st_warning = True)
def detect_emotion_ekman(review):
    ekman = pipeline('sentiment-analysis', model='arpanghoshal/EkmanClassifier')
    result = ekman(review)
    return result[0]

review_id = st.number_input("Specify the review index (0-500)",
                            min_value=0,
                            max_value=500,
                            step=1)

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_user_analysis_data(data, review_id):
    raw_review = data.review[review_id]
    review = data.preprocessed_reviews[review_id]
    bert_sentiment = analyze_sentiment_bert(review)
    ekman_emotion = detect_emotion_ekman(review)
    return raw_review, bert_sentiment, ekman_emotion

if st.button("Analyze"):
    if review_id >= 0 and review_id <= 500:
        raw_review, bert_sentiment, ekman_emotion = get_user_analysis_data(data, review_id)

        sentiment_color = "green" if bert_sentiment["label"] == "POSITIVE" else "red"

        st.write(f"Review: {raw_review}")
        st.write(f"BERT Sentiment Analysis: <span style='color:{sentiment_color}'>{bert_sentiment['label']}</span> ,(Score: {bert_sentiment['score']:.2f})",
                 unsafe_allow_html=True)
        st.write(f"Ekman Emotion Detection: {ekman_emotion['label']} ,(Score: {ekman_emotion['score']:.2f})")
    else:
        st.error("Review index is out of valid range (0-500)")





