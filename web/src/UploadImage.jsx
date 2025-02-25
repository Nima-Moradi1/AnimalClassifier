import { useState } from "react";
import * as tf from "@tensorflow/tfjs";
import PropTypes from "prop-types";

const UploadImage = ({ onResult }) => {
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [model, setModel] = useState(null); // To store the model

  // Load the model once when the component mounts

  const loadModel = async () => {
    try {
      const loadedModel = await tf.loadLayersModel("/web_model/model.json");
      console.log("Model Loaded:", loadedModel);
      console.log("Input Layer:", loadedModel.input);
      console.log("Input Shape:", loadedModel.input.shape);

      setModel(loadedModel);
      console.log("Model loaded successfully");
    } catch (err) {
      console.error("Error loading model:", err);
      alert("Error loading model. Please try again.");
    }
  };

  // Run the model prediction when the user clicks "Predict"
  const handlePredict = async () => {
    if (!image) return alert("Please upload an image first.");
    setLoading(true);
    try {
      const model = await tf.loadLayersModel("/web_model/model.json"); // This assumes that model.json is accessible at this path

      console.log("Model loaded successfully");

      const imgElement = document.createElement("img");
      imgElement.src = image;

      imgElement.onload = async () => {
        const tensor = tf.browser
          .fromPixels(imgElement)
          .resizeBilinear([128, 128]) // Resize the image to match the model's input shape
          .expandDims(0) // Add batch dimension
          .div(255.0); // Normalize the image

        const prediction = model.predict(tensor);
        const labelIndex = prediction.argMax(1).dataSync()[0];
        const labels = ["Cat", "Dog", "Fox"];
        onResult(labels[labelIndex]);

        setLoading(false);
      };
    } catch (err) {
      console.error("Error during model loading or prediction:", err);
      alert("Error loading or predicting with the model. Please try again.");
      setLoading(false);
    }
  };

  // Handle image upload
  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => setImage(reader.result);
      reader.readAsDataURL(file);
    }
  };

  // Load the model when the component mounts
  useState(() => {
    loadModel();
  }, []);

  return (
    <div className="flex flex-col items-center">
      <div>
        <input
          type="file"
          onChange={handleImageUpload}
          className="mb-4"
        />
        {image && (
          <img
            src={image}
            alt="Uploaded"
            className="max-w-md mb-4"
          />
        )}
      </div>
      <button
        onClick={handlePredict}
        className={` ${!image ? "bg-opacity-50" : ""} bg-blue-500 text-white px-4 py-2 rounded-md`}
        disabled={loading || !image || !model}>
        {loading ? "Predicting..." : "Predict"}
      </button>
    </div>
  );
};

// Prop validation
UploadImage.propTypes = {
  onResult: PropTypes.func.isRequired,
};

export default UploadImage;
