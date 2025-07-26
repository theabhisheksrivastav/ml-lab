import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



import unittest
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from lib.utils.model_io import save_model, load_model

class TestModelIO(unittest.TestCase):
    def setUp(self):
        # Use a small sample model
        self.model = MultinomialNB()
        data = load_iris()
        X_train, _, y_train, _ = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        self.path = "models/logistic_model.joblib"

    def test_save_and_load(self):
        save_model(self.model, self.path)
        self.assertTrue(os.path.exists(self.path), "Model file was not created.")
        
        loaded_model = load_model(self.path)
        self.assertIsNotNone(loaded_model, "Loaded model is None.")
        self.assertEqual(type(loaded_model), type(self.model), "Model type mismatch.")

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

if __name__ == '__main__':
    unittest.main()
