from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
print("Current working directory:", os.getcwd())


print("Script is running...")  # Add this as the very first line
app = FastAPI()
print("FastAPI app initialized...")  # Add this right after app initialization
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class_map = {"dogs": 0, "cats": 1, "foxes": 2}


model_path = "model/modified_model3.h5"
print("Initializing FastAPI application...")
model = load_model(model_path)
print("Model loaded successfully")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # فایل هایی که اپلود میشه رو اینجا ذخیره میکنم که داخل لوکال قابل دسترسی باشه
        upload_dir = "static/uploaded_images"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # پردازش تصویر اپلود شده رو اینجا انجام میدم با تنسورفلو
        image = load_img(file_path, target_size=(128, 128))  # Use the same target size as in your script
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # حالا با مدلی که ترین کردم پیش بینی اش رو انجام میدم
        predictions = model.predict(image)
        class_idx = np.argmax(predictions)
        confidence = np.max(predictions)

        class_name = list(class_map.keys())[list(class_map.values()).index(class_idx)]
        
        # اینجا براش یه استانه تعریف کردم که بر اساس دقت بیاد پیش بینی رو اعلام کنه
        if confidence > 0.80:
            result = {"class": class_name, "confidence": float(confidence)}
        else:
            result = {"error": "Cat, dog, or fox not recognized.", "confidence": float(confidence)}

        return result

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)