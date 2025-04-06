from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("plant_disease_model2.h5")

# Class names (must match model output order)
class_names = [ 
   "Pepper__bell___Bacterial_spot",
   "Pepper__bell___healthy",
   "Potato___Early_blight",
   "Potato___Late_blight",
   "Potato___healthy",
   "Tomato_Bacterial_spot",
   "Tomato_Early_blight",
   "Tomato_Late_blight",
   "Tomato_Leaf_Mold",
   "Tomato_Septoria_leaf_spot",
   "Tomato_Spider_mites_Two_spotted_spider_mite",
   "Tomato_Target_Spot",
   "Tomato_Tomato_mosaic_virus",
   "Tomato_Tomato_YellowLeaf_Curl_Virus",
   "Tomato_healthy",
   "Tomato_Bacterial_spot1",
   "Tomato_Early_blight1",
   "Tomato_Late_blight1",
   "Tomato_Leaf_Mold1",
   "Tomato_Tomato_mosaic_virus1",
   "Tomato_Tomato_YellowLeaf_Curl_Virus1",
   "Tomato_healthy1",
   "Pepper__bell___Bacterial_spot1",
   "Pepper__bell___healthy1",
   "Potato___Early_blight1",
   "Potato___healthy1"
]

# Remedies (can tweak later)
remedies = {
    "Pepper__bell___Bacterial_spot": "Use copper-based fungicide and avoid overhead watering.",
    "Pepper__bell___healthy": "No action needed. Keep monitoring.",
    "Potato___Early_blight": "Apply fungicide like chlorothalonil. Remove affected leaves.",
    "Potato___Late_blight": "Use certified seed potatoes and apply appropriate fungicide.",
    "Potato___healthy": "Plant in well-drained soil and rotate crops.",
    "Tomato_Bacterial_spot": "Avoid working with wet plants and apply copper spray.",
    "Tomato_Early_blight": "Remove infected leaves, use mulch, and spray fungicide.",
    "Tomato_Late_blight": "Destroy infected plants, and spray fungicide with mancozeb.",
    "Tomato_Leaf_Mold": "Increase air circulation and apply fungicide.",
    "Tomato_Septoria_leaf_spot": "Use fungicides and avoid overhead watering.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Use neem oil or insecticidal soap.",
    "Tomato_Target_Spot": "Apply preventive fungicides like azoxystrobin.",
    "Tomato_Tomato_mosaic_virus": "Remove infected plants and disinfect tools.",
    "Tomato_Tomato_YellowLeaf_Curl_Virus": "Control whiteflies and remove infected plants.",
    "Tomato_healthy": "Your plant is healthy! Maintain proper care.",
    "Tomato_Bacterial_spot1": "Same as 'Tomato_Bacterial_spot'. Use copper spray.",
    "Tomato_Early_blight1": "Same as 'Tomato_Early_blight'. Apply fungicide.",
    "Tomato_Late_blight1": "Same as 'Tomato_Late_blight'. Remove infected plants.",
    "Tomato_Leaf_Mold1": "Same as 'Tomato_Leaf_Mold'. Increase airflow.",
    "Tomato_Tomato_mosaic_virus1": "Same as before. Destroy infected plants.",
    "Tomato_Tomato_YellowLeaf_Curl_Virus1": "Same as original. Monitor whiteflies.",
    "Tomato_healthy1": "No disease detected. Keep plant healthy.",
    "Pepper__bell___Bacterial_spot1": "Same as original. Use fungicide.",
    "Pepper__bell___healthy1": "Healthy plant. No treatment needed.",
    "Potato___Early_blight1": "Same remedy. Use fungicide and remove bad leaves.",
    "Potato___healthy1": "Healthy plant. Water properly and rotate crops."
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    remedy = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            # Make sure static folder exists
            if not os.path.exists("static"):
                os.makedirs("static")
            image_path=file.filename
            file_path = os.path.join("static", file.filename)
            file.save(file_path)

            # Preprocess image
            img = image.load_img(file_path, target_size=(180, 180))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            pred = model.predict(img_array)
            predicted_class = class_names[np.argmax(pred)]
            confidence = float(np.max(pred)) * 100

            # Get remedy
            remedy = remedies.get(predicted_class, "No remedy found.")
            prediction = f"{predicted_class} ({confidence:.2f}%)"
            image_path=file.filename

    return render_template("index.html", prediction=prediction, remedy=remedy,image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=10000)
