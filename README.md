# Loan Default Prediction App

A full-stack application for predicting loan defaults using Machine Learning.

## Project Structure
-   `backend/`: FastAPI server processing predictions.
-   `frontend/`: Next.js web interface for user interaction.
-   `data/`: Dataset used for training.
-   `scripts/`: Utilities for training models (XGBoost, Neural Network, Logistic Regression).

## Prerequisites
-   **Python 3.10+**
-   **Node.js 18+**

## Setup

### 1. Backend Setup
Install Python dependencies:
```bash
pip install -r requirements.txt
```

### 2. Frontend Setup
Install Node dependencies:
```bash
cd frontend
npm install
cd ..
```

## Running the App

You can run the backend and frontend in separate terminals.

### Terminal 1: Backend
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
The API will be available at `http://localhost:8000`.

### Terminal 2: Frontend
```bash
cd frontend
npm run dev
```
The web app will be available at `http://localhost:3000`.

## (Optional) Training Models

If you wish to retrain the models, you can use the unified training script:

```bash
# Train XGBoost (Default)
python3 scripts/train_model.py --model xgboost

# Train Neural Network
python3 scripts/train_model.py --model neural_network
```
