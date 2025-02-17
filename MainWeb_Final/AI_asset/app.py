from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)  

# Load the CSV data
image_df = pd.read_csv("sorted_food_images_by_name.csv")

# Initialize and fit the label encoder
label_encoder = LabelEncoder()
label_encoder.fit(image_df['Label'])

# Load the full model
def load_model():
    model = torch.load("densenet201_food_classifier_full.pth", map_location=torch.device('cpu'))
    model.eval()  # Set the model to evaluation mode
    return model

model = load_model()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file:
        return jsonify({"error": "No file found"}), 400

    try:
        # Load and preprocess image
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        # Get prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()

        # Map predicted class index to label (using label_encoder)
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]

        # Retrieve calorie information from image_df
        row = image_df[image_df['Label'] == predicted_label].iloc[0]
        calories_kcal = int(row['Calories_kcal'])  # Convert to int
        calories_kJ = int(row['Calories_kJ'])  # Convert to int

        return jsonify({
            "class": predicted_label,
            "calories_kcal": calories_kcal,
            "calories_kJ": calories_kJ,
            "note": "The calorie information provided is an estimation based on the dataset and may not be accurate for all servings or preparations."
        })

    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")  # Log the error
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
