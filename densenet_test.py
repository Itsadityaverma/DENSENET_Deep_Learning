import os
from keras.preprocessing import image
from keras.models import load_model
import numpy as np

# Load the trained model
model = load_model('/Users/macpro/Documents/PLANT DETECTION/KAGGLE/DENSENET/densenet_training_model.h5')

# Path to the directory containing test images
test_dir = '/Users/macpro/Documents/PLANT DETECTION/KAGGLE/test'

# Supported image extensions
supported_extensions = {'.jpg', '.jpeg', '.png'}

# Assuming you have 11 classes, provide the names in the same order as the model's output
class_names = ["Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold", "Septoria_leaf_spot", 
                "Spider_mites_Two-spotted_spider_mite", "Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus", 
                "Tomato_mosaic_virus", "healthy", "powdery_mildew"]

# Loop through each file in the directory
for filename in os.listdir(test_dir):
    # Ensure the file is an image with a supported extension
    if any(filename.lower().endswith(ext) for ext in supported_extensions):
        # Construct the full path to the image
        img_path = os.path.join(test_dir, filename)

        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the pixel values

        # Make predictions
        predictions = model.predict(img_array)

        # Get the predicted class index
        predicted_class_index = np.argmax(predictions)

        # Get the predicted class name
        predicted_class_name = class_names[predicted_class_index]

        # Print the result
        print(f"Image: {filename}, Predicted class name: {predicted_class_name}")
