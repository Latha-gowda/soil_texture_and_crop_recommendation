import tensorflow as tf
from PIL import Image
import numpy as np

# Load the Keras model
model = tf.keras.models.load_model('C:\\Users\\latha\\PycharmProjects\\hackothon\\my_model.keras')

# Load and preprocess the image
img = Image.open('black.jpg').resize((224, 224))
img = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]
img = np.expand_dims(img, axis=0)  # Add batch dimension

# Perform inference
predictions = model.predict(img)

# Output is a probability distribution; get the class with the highest probability
predicted_class = np.argmax(predictions, axis=1)[0]

# List of soil texture classes
classes = ['Alluvial Soil', 'Black Soil', 'Clay Soil', 'Red Soil']

# Get the predicted soil texture
predicted_texture = classes[predicted_class]
print(f"Predicted Soil Texture: {predicted_texture}")

# Crop recommendations based on soil texture
crop_recommendations = {
    'black soil': [
        'Cotton', 'Tobacco', 'Millets', 'Citrus Fruits', 'Maize', 'Sunflower',
        'Castor', 'Oilseeds', 'Jowar', 'Linseed', 'Safflower', 'Sugarcane', 'Vegetables'
    ],
    'clay soil': [
        'Cabbage', 'Kale', 'Spinach', 'Broccoli', 'Brussels Sprouts', 'Cauliflower',
        'Apple Trees', 'Pear Trees', 'Asparagus'
    ],
    'alluvial soil': [
        'Rice', 'Wheat', 'Sugarcane', 'Cotton', 'Maize', 'Barley', 'Groundnut',
        'Soybean', 'Mustard', 'Vegetables', 'Fruits'
    ],
    'red soil': [
        'Groundnut', 'Pulses', 'Millet', 'Cotton', 'Tobacco'
    ]
}

def recommend_crops(predicted_texture):
    # Standardize the input to lowercase and remove leading/trailing spaces
    standardized_texture = predicted_texture.strip().lower()

    # Retrieve crop recommendations based on soil texture
    return crop_recommendations.get(standardized_texture,
                                   "Soil texture not recognized. Please enter 'clay', 'alluvial', 'black', or 'red'.")

# Get crop recommendations
recommended_crops = recommend_crops(predicted_texture)

# Display the results
print(f"Predicted Soil Texture: {predicted_texture}")
if isinstance(recommended_crops, list):
    print("Recommended crops:")
    for crop in recommended_crops:
        print(f"- {crop}")
else:
    print(recommended_crops)
