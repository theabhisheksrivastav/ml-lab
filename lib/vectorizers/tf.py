import math

def compute_tf(tokens, mode='raw'):
    """
    Compute term frequency for a list of tokens.

    Modes:
    - 'raw'        → Basic count
    - 'normalized' → Frequency / total terms
    - 'log'        → 1 + log10(freq)
    - 'binary'     → 1 if word appears, 0 otherwise
    - 'sublinear'  → log(1 + freq) [so 0 freq still gives 0]

    Returns:
    - dict: {token: tf_value}
    """
    tf = {}
    total_tokens = len(tokens)

    # Step 1: Count frequencies
    for token in tokens:
        tf[token] = tf.get(token, 0) + 1

    # Step 2: Apply selected transformation
    if mode == 'normalized' and total_tokens > 0:
        tf = {token: count / total_tokens for token, count in tf.items()}
    elif mode == 'log':
        tf = {token: 1 + math.log10(count) for token, count in tf.items()}
    elif mode == 'binary':
        tf = {token: 1 for token in tf}
    elif mode == 'sublinear':
        tf = {token: math.log(1 + count) for token, count in tf.items()}

    return tf
