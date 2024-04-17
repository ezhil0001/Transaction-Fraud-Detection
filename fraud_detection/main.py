import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def test_model_with_text_only(text_input, model, vectorizer):

    # Make predictions using the trained model
    text_input_tfidf = vectorizer.transform([text_input])
    prediction = model.predict(text_input_tfidf)[0]

    # Map the numerical prediction to the corresponding label
    if prediction == 1:
        return 'fraudulent transaction  🚫'
    else:
        return 'legitimate transaction 💲'

# Load the model and vectorizer
model = joblib.load("fraud.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit app
st.title("Transaction Fraud Detection 🌐💸")

# User input for testing
user_input = st.text_area("Enter the transaction details:",value="An user Donated $1483.31 to a charity. (Category: Big Spending)")

# Button to trigger prediction
if st.button("Predict 💵"):
    if user_input:
        prediction = test_model_with_text_only(user_input, model, vectorizer)
        st.success(f"Prediction : {prediction}")
    else:
        st.warning("Please enter transaction details.")
