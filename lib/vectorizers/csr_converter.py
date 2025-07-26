import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

def to_csr(tfidf_vectors, vocab=None):
    """
    Converts list of TF-IDF dictionaries to CSR matrix.
    
    Args:
        tfidf_vectors (List[Dict[str, float]]): List of TF-IDF dicts per doc.
        vocab (List[str], optional): If provided, fixes feature order.

    Returns:
        csr_matrix, feature_names
    """
    if vocab is None:
        # Build vocab from all terms in all documents
        vocab = sorted(set(term for vec in tfidf_vectors for term in vec))

    vocab_index = {term: idx for idx, term in enumerate(vocab)}
    
    data, row_indices, col_indices = [], [], []
    
    for row, tfidf_dict in enumerate(tfidf_vectors):
        for term, value in tfidf_dict.items():
            col = vocab_index.get(term)
            if col is not None:
                row_indices.append(row)
                col_indices.append(col)
                data.append(value)
    
    matrix = csr_matrix((data, (row_indices, col_indices)), shape=(len(tfidf_vectors), len(vocab)))
    return matrix, vocab

from scipy.sparse import csr_matrix

def to_csr_single(tfidf_dict, feature_names):
    row = []
    col = []
    data = []

    for term, val in tfidf_dict.items():
        if term in feature_names:
            row.append(0)  # row index is always 0 (only one row)
            col.append(feature_names.index(term))  # column index of the term
            data.append(val)  # tf-idf value of the term

    return csr_matrix((data, (row, col)), shape=(1, len(feature_names)))
