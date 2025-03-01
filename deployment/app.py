import torch
import pickle
from flask import Flask, request, jsonify

# Flask App Initialization
app = Flask(__name__)

# Define the GNN Model Class
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = torch.nn.Linear(47, 16)  # Input size: 47 features
        self.conv2 = torch.nn.Linear(16, 2)   # Output size: 2 classes

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        return x

# Load the Model
try:
    with open("gnn_model.pkl", "rb") as file:
        model_state_dict = pickle.load(file)
        print("Checkpoint Loaded")

    model = GNN()
    model.load_state_dict(model_state_dict, strict=False)  # Allow partial loading
    model.eval()
    print("Model successfully loaded and ready for inference")

except Exception as e:
    print(f"Error loading model: {e}")

# API Endpoint for Prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = data["features"]

        if len(features) != 47:
            return jsonify({"error": "Input feature size must be 47"}), 400

        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            prediction = model(features_tensor)
            predicted_class = torch.argmax(prediction, dim=1).item()
        
        return jsonify({"prediction": predicted_class})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
