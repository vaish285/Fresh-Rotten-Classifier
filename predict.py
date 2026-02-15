import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# --------------------------
# SETTINGS
# --------------------------
model_path = "fruit_veg_classifier.h5"
img_size = (224, 224)

# --------------------------
# LOAD MODEL
# --------------------------
model = load_model(model_path)
print("✅ Model loaded successfully")

# --------------------------
# GET CLASS NAMES
# --------------------------
dataset_small = "dataset_small"
class_names = sorted([d for d in os.listdir(dataset_small)
                      if os.path.isdir(os.path.join(dataset_small, d))])
print("Detected classes:", class_names)

# --------------------------
# FUNCTION TO PREDICT ONE IMAGE
# --------------------------
def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        return
    
    img = image.load_img(img_path, target_size=img_size)
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    
    pred = model.predict(x)
    class_index = np.argmax(pred)
    class_name = class_names[class_index]
    confidence = pred[0][class_index] * 100
    
    print(f"Prediction: {class_name} ({confidence:.2f}%)")
    return class_name, confidence

# --------------------------
# FUNCTION TO PREDICT MULTIPLE IMAGES IN A FOLDER
# --------------------------
def predict_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return
    
    images = [f for f in os.listdir(folder_path)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not images:
        print("❌ No images found in the folder")
        return
    
    for img_file in images:
        img_path = os.path.join(folder_path, img_file)
        predict_image(img_path)

# --------------------------
# EXAMPLES
# --------------------------
# Predict one image:
# Replace 'test_image.jpg' with your image file
test_image = "test_image.jpg"
predict_image(test_image)

# Predict all images in a folder:
# Replace 'new_images' with your folder containing multiple images
# test_folder = "new_images"
# predict_folder(test_folder)