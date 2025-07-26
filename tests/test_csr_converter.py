import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import unittest
from lib.vectorizers.csr_converter import to_csr
from scipy.sparse import csr_matrix

class TestCSRConverter(unittest.TestCase):
    def test_basic_csr_conversion(self):
        tfidf_vectors = [
            {"hello": 0.5, "world": 1.0},
            {"world": 0.8, "test": 0.6}
        ]
        matrix, vocab = to_csr(tfidf_vectors)

        self.assertIsInstance(matrix, csr_matrix)
        self.assertEqual(matrix.shape[0], 2)
        self.assertTrue("hello" in vocab)
        self.assertTrue("test" in vocab)

    def test_empty_input(self):
        matrix, vocab = to_csr([])
        self.assertEqual(matrix.shape[0], 0)
        self.assertEqual(matrix.shape[1], 0)
        self.assertEqual(vocab, [])

if __name__ == "__main__":
    unittest.main()

