from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Loan Default Predictor")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model Pipeline
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models/xgboost_pipeline.pkl')

try:
    if os.path.exists(MODEL_PATH):
        pipeline = joblib.load(MODEL_PATH)
        print(f"Model pipeline loaded from {MODEL_PATH}")
    else:
        print(f"Error: Model not found at {MODEL_PATH}")
        pipeline = None
except Exception as e:
    print(f"Error loading model: {e}")
    pipeline = None
    
class LoanApplication(BaseModel):
    Age: int
    Income: int
    LoanAmount: int
    CreditScore: int
    MonthsEmployed: int
    NumCreditLines: int
    InterestRate: float
    LoanTerm: int
    DTIRatio: float
    Education: str
    EmploymentType: str
    MaritalStatus: str
    HasMortgage: str
    HasDependents: str
    LoanPurpose: str
    HasCoSigner: str

@app.get("/")
def read_root():
    return {"message": "Loan Default Prediction API is running", "model_status": "loaded" if pipeline else "failed"}

@app.post("/predict")
def predict_default(application: LoanApplication):
    if not pipeline:
         raise HTTPException(status_code=503, detail="Model not loaded")
         
    try:
        # Convert dictionary to DataFrame
        data = application.model_dump()
        df = pd.DataFrame([data])
        
        # Predict using Pipeline (Handles Preprocessing + Prediction)
        # Probabilities: [prob_0, prob_1]
        probs = pipeline.predict_proba(df)
        default_prob = float(probs[0][1])
        prediction = int(probs[0][1] > 0.5)
        
        return {
            "prediction": prediction,
            "probability": default_prob,
            "status": "High Risk" if prediction == 1 else "Low Risk"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # If running directly, assume we are in backend/
    uvicorn.run(app, host="0.0.0.0", port=8000)
