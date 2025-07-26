import re

def tokenize(text):
    """
    Tokenizes input text into words, hashtags, emoticons, and URLs.
    """
    # Regex parts
    emoticons = r'[:;=8][\-o\*\']?[\)\]\(dDpP/:\}\{@\|\\]'
    hashtags = r'#\w+'
    urls = r'https?://\S+|www\.\S+'
    words = r'\b\w+\b'

    # Combine all patterns
    pattern = f'({urls})|({emoticons})|({hashtags})|({words})'

    tokens = re.findall(pattern, text)

    # `re.findall` returns tuples due to multiple groups, we flatten it
    flat_tokens = [token for group in tokens for token in group if token]

    return flat_tokens
