import uvicorn
from fastapi import FastAPI,File,UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import keras

app = FastAPI()

model = tf.keras.models.load_model("my_model.keras")

@app.get("/")
def home():
    return{"message":"soil_prediction using image"}

@app.post("/predict")
async def predict_soil(file: UploadFile = File(...)):
    try:
        contents = await  file.read()
        image = Image.open(io.BytesIO(contents))
        image = image.resize(128,128)
        image = np.array(image) /255.0
        image = np.expand_dims(image,axis = 0)

        predictions = model.predict(image)
        predicted_class = np.argmax(predictions,axis =1)[0]
        confidence = float(np.max(predictions))

        return {"predicted class": int(predicted_class),"confidence": confidence}

    except Exception as e:
        return {"error":str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


