import unittest
import pickle
import numpy as np
from app import app  # Import the Flask app

class TestModel(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Load the model
        with open('model.pkl', 'rb') as f:
            cls.model = pickle.load(f)

    def test_model_output_shape(self):
        # Test that the model's prediction has the correct shape
        test_data = np.array([5.1, 3.5, 1.4, 0.2]).reshape(1, -1)
        prediction = self.model.predict(test_data)
        self.assertEqual(prediction.shape, (1,))  # Expecting one prediction

    def test_model_prediction(self):
        # Check if the model's prediction is within a valid range
        test_data = np.array([5.1, 3.5, 1.4, 0.2]).reshape(1, -1)
        prediction = self.model.predict(test_data)
        self.assertIn(prediction[0], [0, 1, 2])  # Assume classes are 0, 1, 2


class TestFlaskAPI(unittest.TestCase):
    
    def setUp(self):
        # Set up the test client
        self.app = app.test_client()
        self.app.testing = True

    def test_predict_endpoint(self):
        # Test the /predict API endpoint
        payload = {
            "features": [5.1, 3.5, 1.4, 0.2]
        }
        response = self.app.post('/predict', json=payload)

        # Assert response status code
        self.assertEqual(response.status_code, 200)

        # Assert the response data is in JSON format
        data = response.get_json()
        self.assertIn('prediction', data)

    def test_invalid_input(self):
        # Test invalid input to the API
        payload = {
            "features": [5.1]  # Invalid input (not enough features)
        }
        response = self.app.post('/predict', json=payload)

        # Assert response status code for bad input
        self.assertEqual(response.status_code, 400)


if __name__ == '__main__':
    unittest.main()
