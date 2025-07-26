import random

def load_amazon_reviews(path, sample_size=50000):
    """
    Reads fastText format review dataset, returns sampled texts and labels.
    """
    texts = []
    labels = []
    with open(path, 'r', encoding='utf‑8') as f:
        for line in f:
            if line.startswith("__label__1"):
                label = "neg"
                review = line.split(" ", 1)[1].strip()
            elif line.startswith("__label__2"):
                label = "pos"
                review = line.split(" ", 1)[1].strip()
            else:
                continue
            texts.append(review)
            labels.append(label)
    # Shuffle and sample
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    combined = combined[:sample_size]
    return zip(*combined)
