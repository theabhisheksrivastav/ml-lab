import re
import emoji
import contractions
from textblob import TextBlob
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text(text, correct_spelling=False):
    # Expand contractions: don't -> do not
    text = contractions.fix(text)

    # Convert emojis to words: ❤️ → red heart
    text = emoji.demojize(text, delimiters=(" ", " "))

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove digits
    text = re.sub(r'\d+', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize and remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]

    # Spelling correction (optional: slow for large texts)
    if correct_spelling:
        corrected = []
        for word in tokens:
            corrected_word = str(TextBlob(word).correct())
            corrected.append(corrected_word)
        tokens = corrected

    return  ' '.join(tokens)
    