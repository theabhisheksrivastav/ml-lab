from lib.vectorizers.tf import compute_tf
if __name__ == "__main__":
    tokens = ['good', 'movie', 'movie', 'excellent', 'movie', 'good']

    for mode in ['raw', 'normalized', 'log', 'binary', 'sublinear']:
        print(f"\nTF ({mode}):")
        print(compute_tf(tokens, mode=mode))
