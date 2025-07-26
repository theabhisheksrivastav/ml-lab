import pandas as pd
from scipy.sparse import save_npz

def save_as_npz(matrix, filename="tfidf_matrix.npz"):
    save_npz(filename, matrix)

def save_as_csv(matrix, vocab, filename="tfidf_matrix.csv"):
    df = pd.DataFrame.sparse.from_spmatrix(matrix, columns=vocab)
    df.to_csv(filename, index=False)
