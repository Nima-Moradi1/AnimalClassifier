import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def check_images(data_dir):
    """Check if all images are valid."""
    valid_files = []
    classes = os.listdir(data_dir)
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        for img_file in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_file)
            try:
                with Image.open(img_path) as img:
                    img.verify()  # اول چک میکنیم که عکس قابل دریافت باشه
                valid_files.append((img_path, cls))
            except Exception as e:
                print(f"Invalid image: {img_path} - {e}")
    return valid_files

def prepare_data(data_dir, image_size=(128, 128), test_size=0.2):
    """Load images, resize them, ensure RGB, and split into train/test sets."""
    valid_files = check_images(data_dir)
    images, labels = [], []
    class_map = {}
    for img_path, cls in valid_files:
        if cls not in class_map:
            class_map[cls] = len(class_map)
        try:
            # Open the image and convert to RGB
            image = Image.open(img_path).convert("RGB").resize(image_size)
            images.append(np.array(image) / 255.0)  # Normalize pixel values
            labels.append(class_map[cls])
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
    images = np.array(images)  # Convert list to NumPy array
    labels = np.array(labels)
    return train_test_split(images, labels, test_size=test_size, random_state=42), class_map

if __name__ == "__main__":
    data_dir = "data/"
    (X_train, X_test, y_train, y_test), class_map = prepare_data(data_dir)
    np.save("scripts/X_train.npy", X_train)
    np.save("scripts/X_test.npy", X_test)
    np.save("scripts/y_train.npy", y_train)
    np.save("scripts/y_test.npy", y_test)
    print("Data preprocessing complete. Class mapping:", class_map)