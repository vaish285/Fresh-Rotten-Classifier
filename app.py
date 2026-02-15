from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import webbrowser
import threading
import shutil

# --------------------------
# SETTINGS
# --------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # must be inside static/ to serve images

model_path = "fruit_veg_classifier.h5"
img_size = (224, 224)

# Load model
model = load_model(model_path)
print("✅ Model loaded successfully")

# Get class names from dataset_small
dataset_small = "dataset_small"
class_names = sorted([d for d in os.listdir(dataset_small)
                      if os.path.isdir(os.path.join(dataset_small, d))])
print("Detected classes:", class_names)

# --------------------------
# FUNCTION TO PREDICT
# --------------------------
def predict_image(img_path):
    img = image.load_img(img_path, target_size=img_size)
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x)
    class_index = np.argmax(pred)
    class_name = class_names[class_index]
    confidence = pred[0][class_index] * 100
    return class_name, confidence

# --------------------------
# ROUTES
# --------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    img_url = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction="No file uploaded", img_url=None)
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction="No selected file", img_url=None)

        # Save uploaded image in static/uploads/
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(img_path)
        img_url = f"{img_path}"  # for HTML display

        # Predict
        class_name, confidence = predict_image(img_path)
        prediction = f"{class_name} ({confidence:.2f}%)"

    return render_template('index.html', prediction=prediction, img_url=img_url)

# --------------------------
# FUNCTION TO OPEN BROWSER AUTOMATICALLY
# --------------------------
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

# --------------------------
# RUN APP
# --------------------------
if __name__ == '__main__':
    print("✅ Starting Flask app...")
    threading.Timer(1, open_browser).start()  # open browser automatically
    app.run(debug=True, host='0.0.0.0', port=5000)