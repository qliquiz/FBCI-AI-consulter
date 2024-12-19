import json
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer

model = tf.keras.models.load_model('faq_model.h5')
with open('faq_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

answers = [item['answer'] for item in data]
vectorizer = CountVectorizer()
vectorizer.fit([item['question'] for item in data])

print("Добро пожаловать в систему вопрос-ответ! Введите 'exit' для выхода.")
while True:
    user_question = input("\nВаш вопрос: ")
    if user_question.lower() == 'exit':
        print("Спасибо за использование системы!")
        break

    question_vector = vectorizer.transform([user_question]).toarray()

    prediction = model.predict(question_vector)
    answer_index = np.argmax(prediction)

    print("\nОтвет:", answers[answer_index])
