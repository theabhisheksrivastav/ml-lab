import random
import nltk

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.utils.model_io import save_model

from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from lib.preprocessing.clean_text import clean_text
from lib.preprocessing.lemmatizer import lemmatize_tokens
from lib.preprocessing.tokenizer import tokenize
from lib.vectorizers.tf import compute_tf
from lib.vectorizers.idf import compute_idf
from lib.vectorizers.tfidf import compute_tfidf
from lib.vectorizers.csr_converter import to_csr

nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


# 1. Load data
documents = [(movie_reviews.raw(fileid), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)
texts = [doc for doc, label in documents]
labels = [label for doc, label in documents]

# 2. Preprocessing
preprocessed = []
for text in texts:
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    lemmatized = lemmatize_tokens(tokens)
    preprocessed.append(lemmatized)

# 3. TF, IDF, TF-IDF
tf_vectors = [compute_tf(preproces) for preproces in preprocessed]
idf_vector = compute_idf(preprocessed)
tfidf_matrix = compute_tfidf(tf_vectors, idf_vector)

# 4. Convert to CSR
X_csr, feature_names = to_csr(tfidf_matrix)
y = labels

# 5. Split
X_train, X_test, y_train, y_test = train_test_split(X_csr, y, test_size=0.2, random_state=42)


# 6. Train + evaluate
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


# 7. Run both
log_model, log_acc = train_and_evaluate(X_train, y_train, X_test, y_test, 'logistic')
nb_model, nb_acc = train_and_evaluate(X_train, y_train, X_test, y_test, 'naive_bayes')

print(f"\nLogistic Regression Accuracy: {log_acc:.4f}")
print(f"Naive Bayes Accuracy:        {nb_acc:.4f}")

# Save models
save_model(log_model, "models/logistic_model.joblib")
save_model(nb_model, "models/naive_bayes_model.joblib")

import joblib
joblib.dump(idf_vector, "models/idf_vector.joblib")
joblib.dump(feature_names, "models/feature_names.joblib")
