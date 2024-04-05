# model.py
import json
import nltk
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def train_model():
    # Load JSON data
    with open('full.json') as file:
        data = json.load(file)

    stemmer = nltk.LancasterStemmer()

    words = []
    classes = []
    documents = []
    ignore_words = ['?']

    for intent in data['intents']:
        for pattern in intent['patterns']:
            # Tokenize each word
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            # Add documents in the corpus
            documents.append((' '.join(w), intent['tag']))
            # Add to our classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))

    # Create training data
    X_train = []
    y_train = []
    for doc in documents:
        bag = [1 if stemmer.stem(word.lower()) in doc[0] else 0 for word in words]
        X_train.append(bag)
        y_train.append(doc[1])

    X_train = np.array(X_train)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y_train)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    y_train = onehot_encoder.fit_transform(integer_encoded)

    # Define and train the neural network model
    model = Sequential()
    model.add(Dense(128, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(classes), activation='softmax'))

    optimizer = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size=8)

    # Save the trained model
    model.save('my_chatbot_model.h5')

if __name__ == "__main__":
    train_model()
