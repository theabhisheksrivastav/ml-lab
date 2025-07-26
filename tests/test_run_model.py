import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
from lib.utils.model_io import load_model
from lib.preprocessing.clean_text import clean_text
from lib.preprocessing.tokenizer import tokenize
from lib.preprocessing.lemmatizer import lemmatize_tokens
from lib.vectorizers.tf import compute_tf
from lib.vectorizers.tfidf import compute_tfidf
from lib.vectorizers.csr_converter import to_csr_single  # <-- added below

# ---------------------
# Load model
# ---------------------
model = load_model("models/amazon_nb.joblib")  # or use "naive_bayes_model.joblib"

# ---------------------
# Load idf_vector & feature_names saved from training
# ---------------------
idf_vector = joblib.load("models/amazon_idf_vec.joblib")
feature_names = joblib.load("models/amazon_features.joblib")

# ---------------------
# Sample test input
# ---------------------
test_text = "T Rex : I check my heart rate from oxymeter and t rex at the same time but t rex 3 disappointed me, And also the temperature is not correct."

# Preprocessing
cleaned = clean_text(test_text)
tokens = tokenize(cleaned)
lemmatized = lemmatize_tokens(tokens)

# Vectorize
tf_vector = compute_tf(lemmatized)
tfidf_vector = compute_tfidf([tf_vector], idf_vector)[0]

# Convert to CSR matrix (single row)
def to_csr_single(tfidf_dict, feature_names):
    from scipy.sparse import csr_matrix

    row = []
    col = []
    data = []

    for term, val in tfidf_dict.items():
        if term in feature_names:
            row.append(0)
            col.append(feature_names.index(term))
            data.append(val)

    return csr_matrix((data, (row, col)), shape=(1, len(feature_names)))

X_test = to_csr_single(tfidf_vector, feature_names)

# Predict
prediction = model.predict(X_test)[0]
print(f"Input: {test_text}")
print(f"Predicted Sentiment: {prediction}")
