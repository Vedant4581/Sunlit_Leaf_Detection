import os
import random
import shutil

# Set seed for reproducibility
random.seed(42)

# Paths
base_dir = 'dataset'
images_dir = os.path.join(base_dir, 'images')
masks_dir = os.path.join(base_dir, 'masks')

# Desired split sizes
train_size = 865
val_size = 223
test_size = 137

# Get all image filenames (without extension)
image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
image_names = [os.path.splitext(f)[0] for f in image_files]

# Shuffle and split
random.shuffle(image_names)
train_names = image_names[:train_size]
val_names = image_names[train_size:train_size + val_size]
test_names = image_names[train_size + val_size:]

# Helper to move files
def move_files(names, split):
    for name in names:
        img_src = os.path.join(images_dir, f"{name}.jpg")
        mask_src = os.path.join(masks_dir, f"{name}_L.png")

        img_dst = os.path.join(images_dir, split, f"{name}.jpg")
        mask_dst = os.path.join(masks_dir, split, f"{name}_L.png")

        shutil.move(img_src, img_dst)
        shutil.move(mask_src, mask_dst)

# Create split folders
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(images_dir, split), exist_ok=True)
    os.makedirs(os.path.join(masks_dir, split), exist_ok=True)

# Move files
move_files(train_names, 'train')
move_files(val_names, 'val')
move_files(test_names, 'test')

print("✅ Dataset split completed.")