import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI()

# Load the trained model
model_path = r"C:\Users\latha\PycharmProjects\hackothon\crop_reccommendation\croprecmodel.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Define input data model
class CropInput(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    temperature: float
    humidity: float
    pH: float
    rainfall: float

# Root Route
@app.get("/")
def home():
    return {"message": "Welcome to the Crop Recommendation API"}

# Prediction Route
@app.post("/predict")
def predict(data: CropInput):
    input_features = np.array([[data.nitrogen, data.phosphorus, data.potassium,
                                data.temperature, data.humidity, data.pH, data.rainfall]])
    prediction = model.predict(input_features)
    return {"Recommended_Crop": prediction[0]}

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
