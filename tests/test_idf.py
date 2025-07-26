from lib.vectorizers.idf import compute_idf


if __name__ == "__main__":
    docs = [
        ["good", "movie", "nice", "acting"],
        ["bad", "acting", "poor", "movie"],
        ["movie", "excellent", "good", "performance"]
    ]

    idf_values = compute_idf(docs)
    print("IDF scores:")
    for word, score in idf_values.items():
        print(f"{word}: {score:.4f}")
