import tensorflow as tf
import pickle

# Load the .keras model
model = tf.keras.models.load_model("my_model.keras")

# Extract model architecture and weights
model_data = {
    "architecture": model.to_json(),
    "weights": model.get_weights()
}

# Save to .pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("Model successfully converted to model.pkl")
