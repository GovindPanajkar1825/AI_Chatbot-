import random
import json
import nltk
from nltk.stem import WordNetLemmatizer
from transformers import pipeline

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Simple intent-based chatbot example
intents = {
    "greeting": ["hello", "hi", "hey"],
    "goodbye": ["bye", "see you", "goodbye"],
    "thanks": ["thanks", "thank you"]
}

# Transformer-based fallback for contextual responses
qa_pipeline = pipeline("question-answering")

def preprocess(sentence):
    return [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence)]

def get_intent(user_input):
    words = preprocess(user_input)
    for intent, keywords in intents.items():
        if any(word in words for word in keywords):
            return intent
    return "fallback"

def get_response(user_input):
    intent = get_intent(user_input)
    if intent == "greeting":
        return random.choice(["Hello!", "Hi there!", "Hey!"])
    elif intent == "goodbye":
        return random.choice(["Goodbye!", "See you later!"])
    elif intent == "thanks":
        return "You're welcome!"
    else:
        # fallback using transformer QA model
        context = "I am a chatbot. I can answer simple questions and help with customer support."
        result = qa_pipeline(question=user_input, context=context)
        return result['answer']
      
