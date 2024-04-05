from flask import Flask, request, jsonify
import json
import random
import nltk
import numpy as np
from keras.models import load_model
from model import train_model
import uuid

app = Flask(__name__)

stemmer = nltk.LancasterStemmer()

# Load JSON data
with open('full.json') as file:
    data = json.load(file)

words = []
classes = []
ignore_words = ['?']

for intent in data['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Load trained model
try:
    model = load_model('my_chatbot_model.h5')
except FileNotFoundError:
    train_model()
    model = load_model('my_chatbot_model.h5')

def predict_class(sentence):
    bag = [1 if stemmer.stem(word.lower()) in sentence else 0 for word in words]
    return model.predict(np.array([bag]))[0]

def get_response(intent_tag):
    for intent in data['intents']:
        if intent['tag'] == intent_tag:
            response = random.choice(intent['responses'])
            # Replace {{booking_id}} with a randomly generated booking ID
            if '{{booking_id}}' in response:
                booking_id = str(uuid.uuid4())[:8]  # Generate a random booking ID
                response = response.replace('{{booking_id}}', booking_id)
            return response

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json['message']
    intent_tag = np.argmax(predict_class(message))
    response = get_response(classes[intent_tag])
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
