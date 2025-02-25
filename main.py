import tensorflow as tf
import numpy as np
import sys
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def predict(image_path, model_path, class_map):
    model = tf.keras.models.load_model(model_path)
    image = load_img(image_path, target_size=(128, 128))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    predictions = model.predict(image)
    class_idx = np.argmax(predictions)
    class_name = list(class_map.keys())[list(class_map.values()).index(class_idx)]
    confidence = np.max(predictions)
    
    if confidence > 0.80:
        return f"Prediction: {class_name} ({confidence:.2f})"
    else:
        return "Error: Cat, dog, or fox not recognized."

if __name__ == "__main__":
    class_map = {"dogs": 0, "cats": 1, "foxes": 2}
    image_path = sys.argv[1]
    print(predict(image_path, "models/modified_model.h5", class_map))