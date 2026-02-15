import os
import random
import shutil

# --------------------------
# Settings
# --------------------------
num_images_per_class = 200  # number of images to copy per class
source_base = "dataset"     # your full dataset folder
target_base = "dataset_small"  # downsized dataset folder

# --------------------------
# Remove old dataset_small if it exists
# --------------------------
if os.path.exists(target_base):
    shutil.rmtree(target_base)
os.makedirs(target_base, exist_ok=True)

# --------------------------
# Process each class folder
# --------------------------
for class_name in os.listdir(source_base):
    source_folder = os.path.join(source_base, class_name)
    target_folder = os.path.join(target_base, class_name)
    os.makedirs(target_folder, exist_ok=True)
    
    # Only take image files
    all_images = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    
    # Randomly select images
    selected_images = random.sample(all_images, min(num_images_per_class, len(all_images)))
    
    # Copy images
    for img in selected_images:
        shutil.copy(os.path.join(source_folder, img), os.path.join(target_folder, img))

print("✅ dataset_small created successfully with all 16 classes!")