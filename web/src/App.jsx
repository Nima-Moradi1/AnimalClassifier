import { useState } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null); // To store the selected file
  const [result, setResult] = useState(null); // To store the prediction result
  const [preview, setPreview] = useState(null); // To store the image preview URL
  const [error, setError] = useState(null); // To store any error

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);
    setResult(null); // Clear previous result when a new file is selected
    setError(null); // Clear any error when a new file is selected
    if (selectedFile) {
      const previewUrl = URL.createObjectURL(selectedFile);
      setPreview(previewUrl);
    } else {
      setPreview(null);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post("http://localhost:8000/predict/", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(response.data);
    } catch (err) {
      setError("Error predicting image. Please try again.");
      console.error(err);
    }
  };

  return (
    <div className="min-h-screen bg-slate-300 flex flex-col justify-center items-center gap-8">
      <h1 className="text-5xl">Animal Classifier</h1>
      <form
        onSubmit={handleSubmit}
        className="flex flex-col gap-5">
        <label
          htmlFor="fileInput"
          className={` ${
            file ? "bg-green-500" : ""
          } "cursor-pointer w-40 text-center bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600" `}>
          {file ? "File Selected ✅" : "Choose File"}
        </label>
        <input
          id="fileInput"
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="hidden"
        />
        {preview && (
          <img
            src={preview}
            className="max-w-md m-3 rounded-lg shadow-sm shadow-black"
          />
        )}
        <button
          disabled={!file || result !== null || error}
          className={`${
            !file || result != null ? "bg-gray-400" : result?.error != null ? "bg-red-400" : ""
          } py-2 px-4 rounded bg-blue-500 text-white`}
          type="submit">
          {result !== null && result?.error == null ? "Predicted..." : result?.error ? "Error !" : "Predict"}
        </button>
      </form>
      {result && result.class && (
        <div className="flex flex-col gap-10 mb-3">
          <h2 className="text-4xl font-bold italic text-center">Prediction</h2>
          <div className="flex gap-10">
            <p className="capitalize text-xl">
              Predicted Animal:
              <span className="mx-5 font-bold border py-3 px-5 rounded-lg border-green-500">{result.class}</span>
            </p>
            <p className="text-xl">
              Confidence:
              <span className="mx-5 font-bold border py-3 px-5 rounded-lg border-green-500">
                {(result.confidence * 100).toFixed(2)}%
              </span>
            </p>
          </div>
        </div>
      )}
      {result && result.error && <p className="text-red-500 mb-3 text-xl">Error : {result.error}</p>}
    </div>
  );
}

export default App;
