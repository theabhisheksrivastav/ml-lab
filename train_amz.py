from load_amazon_reviews import load_amazon_reviews
from lib.preprocessing.clean_text import clean_text
from lib.preprocessing.tokenizer import tokenize
from lib.preprocessing.lemmatizer import lemmatize_tokens
from lib.vectorizers.tf import compute_tf
from lib.vectorizers.idf import compute_idf
from lib.vectorizers.tfidf import compute_tfidf
from lib.vectorizers.csr_converter import to_csr
from lib.utils.model_io import save_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load data from Amazon reviews
texts, labels = load_amazon_reviews("data/amazonreviews/test.ft.txt", sample_size=60000)

# Preprocess
preproc = [
    lemmatize_tokens(tokenize(clean_text(text)))
    for text in texts
]

# Vectorize
tf_vecs = [compute_tf(doc) for doc in preproc]
idf_vec = compute_idf(preproc)
tfidf_list = compute_tfidf(tf_vecs, idf_vec)

# Build CSR matrix
X_csr, feature_names = to_csr(tfidf_list)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_csr, labels, test_size=0.2, random_state=42)

# Train logistic & NB
lr = LogisticRegression(max_iter=1000); lr.fit(X_train, y_train)
nb = MultinomialNB(); nb.fit(X_train, y_train)

print("Logistic Acc:", accuracy_score(y_test, lr.predict(X_test)))
print("NB Acc:", accuracy_score(y_test, nb.predict(X_test)))

# Save artifacts
save_model(lr, "models/amazon_logistic.joblib")
save_model(nb, "models/amazon_nb.joblib")
joblib.dump(idf_vec, "models/amazon_idf_vec.joblib")
joblib.dump(feature_names, "models/amazon_features.joblib")
