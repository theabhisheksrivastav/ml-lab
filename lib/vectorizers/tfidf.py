def compute_tfidf(tf_scores, idf_scores):
    """
    Computes TF-IDF vectors for each document.

    Parameters:
    - tf_scores: list of dicts → [{term1: tf, term2: tf, ...}, {...}, ...]
    - idf_scores: dict → {term: idf}

    Returns:
    - tfidf_vectors: list of dicts → [{term1: tfidf, term2: tfidf, ...}, {...}, ...]
    """
    tfidf_vectors = []
    for doc_tf in tf_scores:
        doc_vector = {}
        for term, tf in doc_tf.items():
            idf = idf_scores.get(term, 0.0)  # Use 0 if term not found in IDF
            doc_vector[term] = tf * idf
        tfidf_vectors.append(doc_vector)
    return tfidf_vectors
