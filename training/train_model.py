import os
import sys
import random
import pandas as pd
import nltk

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Custom module paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.utils.model_io import save_model
from lib.preprocessing.clean_text import clean_text
from lib.preprocessing.lemmatizer import lemmatize_tokens
from lib.preprocessing.tokenizer import tokenize
from lib.vectorizers.tf import compute_tf
from lib.vectorizers.idf import compute_idf
from lib.vectorizers.tfidf import compute_tfidf
from lib.vectorizers.csr_converter import to_csr

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

def load_dataset(use_custom=False, path=None):
    if use_custom and path:
        print(f"📂 Loading custom dataset from {path}")
        df = pd.read_csv(path)
        texts = df['text'].tolist()
        labels = df['label'].tolist()
    else:
        print("📂 Loading default NLTK movie_reviews dataset")
        from nltk.corpus import movie_reviews
        nltk.download('movie_reviews')
        documents = [(movie_reviews.raw(fileid), category)
                     for category in movie_reviews.categories()
                     for fileid in movie_reviews.fileids(category)]
        random.shuffle(documents)
        texts = [doc for doc, label in documents]
        labels = [label for doc, label in documents]
    return texts, labels


def preprocess_texts(texts):
    preprocessed = []
    for text in texts:
        cleaned = clean_text(text)
        tokens = tokenize(cleaned)
        lemmatized = lemmatize_tokens(tokens)
        preprocessed.append(lemmatized)
    return preprocessed


def train_and_evaluate(X_train, y_train, X_test, y_test, model_type='logistic'):
    if model_type == 'naive_bayes':
        model = MultinomialNB()
    elif model_type == 'logistic':
        model = LogisticRegression(penalty='l2', max_iter=1000)
    else:
        raise ValueError("model_type must be either 'logistic' or 'naive_bayes'")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    return model, acc


def main(custom_csv_path=None):
    # Step 1: Load data
    use_custom = bool(custom_csv_path)
    texts, labels = load_dataset(use_custom, custom_csv_path)

    # Step 2: Preprocess
    preprocessed = preprocess_texts(texts)

    # Step 3: Vectorization
    tf_vectors = [compute_tf(p) for p in preprocessed]
    idf_vector = compute_idf(preprocessed)
    tfidf_matrix = compute_tfidf(tf_vectors, idf_vector)

    # Step 4: CSR Conversion
    X_csr, feature_names = to_csr(tfidf_matrix)
    y = labels

    # Step 5: Split
    X_train, X_test, y_train, y_test = train_test_split(X_csr, y, test_size=0.2, random_state=42)

    # Step 6: Train both models
    log_model, log_acc = train_and_evaluate(X_train, y_train, X_test, y_test, 'logistic')
    nb_model, nb_acc = train_and_evaluate(X_train, y_train, X_test, y_test, 'naive_bayes')

    print(f"\n✅ Logistic Regression Accuracy: {log_acc:.4f}")
    print(f"✅ Naive Bayes Accuracy:        {nb_acc:.4f}")

    # Step 7: Save models and vector data
    os.makedirs("models", exist_ok=True)
    save_model(log_model, "models/logistic_model.joblib")
    save_model(nb_model, "models/naive_bayes_model.joblib")
    joblib.dump(idf_vector, "models/idf_vector.joblib")
    joblib.dump(feature_names, "models/feature_names.joblib")


if __name__ == "__main__":
    # ✅ Set your custom CSV path here
    csv_path = "data/sentiment_dataset.csv"  # ← Change this to your actual file
    main(custom_csv_path=csv_path)
