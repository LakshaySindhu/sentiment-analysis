import os
from flask import Flask, render_template, request, jsonify
import joblib
from preprocess import preprocess_text

app = Flask(__name__, static_url_path='/static')

model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def index():
    return render_template('index.html')

sentiment_to_index = {
    'Negative': 0,
    'Neutral': 1,
    'Positive': 2
}

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')
    preprocessed_text = preprocess_text(text)  
    vectorized_text = vectorizer.transform([preprocessed_text])
    sentiment = model.predict(vectorized_text)[0]
    sentiments = ["Negative", "Neutral", "Positive"]
    result = sentiments[sentiment_to_index[sentiment]]
    return jsonify({'sentiment': result})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
