from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the pretrained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize Flask application
app = Flask(__name__)

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})

# Start the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
