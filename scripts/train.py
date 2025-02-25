import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models

def build_model(input_shape, num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=input_shape
    )
    base_model.trainable = False  # بیس مدل رو فریز میکنم که اپدیت نشه
    input_shape = (128, 128, 3)  # یه اینپوت شیپ دیفالت میدیم برای rgb
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model

if __name__ == "__main__":
    # دیتا رو اینجا پردازش میکنیم 
    X_train = np.load("scripts/X_train.npy")
    X_test = np.load("scripts/X_test.npy")
    y_train = np.load("scripts/y_train.npy")
    y_test = np.load("scripts/y_test.npy")

    # Build and compile the model
    model = build_model(input_shape=X_train.shape[1:], num_classes=3)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

    # Save the model
    model.save("models/modified_model3.h5") # .keras
    model.summary()
    print("Model training complete")