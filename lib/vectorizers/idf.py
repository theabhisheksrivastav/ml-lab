import math
from collections import defaultdict

def compute_idf(documents):
    """
    Compute IDF for a list of documents.

    Parameters:
    - documents (List[List[str]]): Each document is a list of tokens.

    Returns:
    - dict: {term: idf_value}
    """
    N = len(documents)
    df = defaultdict(int)

    # Count document frequency
    for doc in documents:
        unique_tokens = set(doc)
        for token in unique_tokens:
            df[token] += 1

    # Compute IDF with smoothing
    idf = {}
    for term, freq in df.items():
        # idf[term] = math.log((N) / (1 + freq))  # base e log by default
        idf[term] = math.log(1 + (N / (1 + freq)))  # smoothed and always >= 0


    return idf
