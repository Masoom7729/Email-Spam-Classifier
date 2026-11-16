import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords', quiet=True)

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Load saved files
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

model = pickle.load(open('model.pkl', 'rb'))

# Same preprocessing used during training
def transform_text(text):

    text = text.lower()

    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [
        ps.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Website heading
st.title("Email Spam Classifier")

st.write(
    "Enter your message below to check whether "
    "it is Spam or Not Spam."
)

# User input
input_sms = st.text_area("Enter your message")

# Prediction button
if st.button("Predict"):

    if input_sms.strip() == "":
        st.warning("Please enter a message.")

    else:

        # Preprocess input
        transformed_sms = transform_text(input_sms)

        # Convert text into numbers
        vector_input = tfidf.transform([transformed_sms])

        # Make prediction
        prediction = model.predict(vector_input)[0]

        # Display result
        if prediction == 1:
            st.error("This message is SPAM!")

        else:
            st.success("This message is NOT SPAM!")