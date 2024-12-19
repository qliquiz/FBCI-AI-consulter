import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

with open('faq_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

questions = [item['question'] for item in data]
answers = [item['answer'] for item in data]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(questions).toarray()

Y = np.arange(len(answers))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(512, activation='relu', input_dim=X.shape[1]),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(len(answers), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=50, batch_size=16, validation_data=(X_test, Y_test))

model.save('faq_model.h5')