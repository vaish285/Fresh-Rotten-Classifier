import os
import random
import shutil

# --------------------------
# SETTINGS
# --------------------------
num_images_per_class = 200   # number of images per class to copy
source_base = "dataset"      # original dataset folder
target_base = "dataset_small"  # downsized dataset folder

# --------------------------
# Remove old dataset_small if it exists
# --------------------------
if os.path.exists(target_base):
    shutil.rmtree(target_base)
os.makedirs(target_base, exist_ok=True)

# --------------------------
# Loop through all folders in dataset
# --------------------------
for class_name in os.listdir(source_base):
    source_folder = os.path.join(source_base, class_name)
    
    # Only process if it's a folder
    if not os.path.isdir(source_folder):
        print(f"Skipping {source_folder} (not a folder)")
        continue
    
    target_folder = os.path.join(target_base, class_name)
    os.makedirs(target_folder, exist_ok=True)
    
    # List only image files
    all_images = [f for f in os.listdir(source_folder)
                  if os.path.isfile(os.path.join(source_folder, f)) and
                     f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # If folder has no images, skip
    if len(all_images) == 0:
        print(f"No images found in {source_folder}, skipping...")
        continue
    
    # Randomly select images
    selected_images = random.sample(all_images, min(num_images_per_class, len(all_images)))
    
    # Copy selected images to dataset_small
    for img in selected_images:
        shutil.copy(os.path.join(source_folder, img), os.path.join(target_folder, img))

print("✅ dataset_small created successfully with all folders containing images!")