from flask import Flask, request, jsonify, render_template
import joblib
from lib.preprocessing.clean_text import clean_text
from lib.preprocessing.lemmatizer import lemmatize_tokens
from lib.preprocessing.tokenizer import tokenize
from lib.vectorizers.tf import compute_tf
from lib.vectorizers.tfidf import compute_tfidf
from lib.vectorizers.csr_converter import to_csr_single

app = Flask(__name__)

# Load model and preprocessing artifacts
model = joblib.load("models/amazon_nb.joblib")
idf_vector = joblib.load("models/amazon_idf_vec.joblib")
feature_names = joblib.load("models/amazon_features.joblib")

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    lemmatized = lemmatize_tokens(tokens)
    
    tf_vector = compute_tf(lemmatized)                      # ✅ just one document
    tfidf_vector = compute_tfidf([tf_vector], idf_vector)   # ✅ list with one dict
    X_csr = to_csr_single(tfidf_vector[0], feature_names)   # ✅ pass only the dict

    prediction = model.predict(X_csr)[0]
    return render_template('index.html', input=text, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
