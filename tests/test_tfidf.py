import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from lib.vectorizers.tf import compute_tf
from lib.vectorizers.idf import compute_idf
from lib.vectorizers.tfidf import compute_tfidf



docs = [
    ["good", "movie", "nice", "acting"],
    ["bad", "acting", "poor", "movie"],
    ["movie", "excellent", "good", "performance"]
]

tf_scores = [compute_tf(doc) for doc in docs]

idf_scores = compute_idf(docs)
tfidf = compute_tfidf(tf_scores, idf_scores)

print(tfidf)

from lib.vectorizers.csr_converter import to_csr
from lib.vectorizers.save_matrix import save_as_csv, save_as_npz

csr_mat, vocab = to_csr(tfidf)
save_as_npz(csr_mat)
save_as_csv(csr_mat, vocab)

from sklearn.metrics.pairwise import cosine_similarity

sim = cosine_similarity(csr_mat)
print(sim)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Example dummy labels
labels = [1, 0, 1]  # Positive, Negative, Positive

X_train, X_test, y_train, y_test = train_test_split(csr_mat, labels, test_size=0.3)
clf = LogisticRegression()
clf.fit(X_train, y_train)
print("Accuracy:", clf.score(X_test, y_test))
