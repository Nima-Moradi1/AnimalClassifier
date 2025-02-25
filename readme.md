# Animal Classification with FastAPI & Vite.js

This project is a machine learning-based web application that classifies images of animals (dogs, cats, and wolves). The system is built using Python for backend processing and FastAPI for API integration, while the frontend is developed using Vite.js and styled with Tailwind CSS.

## Features
- Upload an image via the web interface
- Backend processes the image and classifies it as a dog, cat, or wolf
- Machine learning model trained with a dataset of animal images
- FastAPI-based REST API for communication between frontend and backend
- Preprocessing script to prepare images before inference
- Trained model included in the project

## Tech Stack
- **Backend:** Python, FastAPI, TensorFlow/PyTorch (for model inference)
- **Frontend:** Vite.js, React.js, Tailwind CSS
- **Model Training:** Custom dataset and training pipeline


## Setup & Installation

### Backend Setup
1. Install dependencies:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```
2. Run FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```
3. API will be available at `http://127.0.0.1:8000`

### Frontend Setup
1. Navigate to the frontend folder:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the frontend:
   ```bash
   npm run dev
   ```
4. The web interface will be available at `http://localhost:5173`

## Model Training
If you want to retrain the model, run:
```bash
python train.py
```
This script will preprocess images, train a deep learning model, and save it in the `/models` directory.

