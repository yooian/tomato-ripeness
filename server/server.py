from flask import Flask, request, jsonify
import os
from datetime import datetime
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array

# Setup
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once when server starts
model = tf.keras.models.load_model("../models/updated_tomato_ripeness_model.keras")
class_names = ['Damaged', 'Old', 'Ripe', 'Unripe']

@app.route('/upload', methods=['POST'])
def upload_and_predict():
    if request.data:
        # Save image
        filename = datetime.now().strftime("img_%Y%m%d_%H%M%S.jpg")
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        with open(filepath, 'wb') as f:
            f.write(request.data)

        # Load and preprocess image
        try:
            img = load_img(filepath, target_size=(256, 256))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Run inference
            predictions = model.predict(img_array)
            predicted_class_idx = int(np.argmax(predictions, axis=1)[0])
            predicted_label = class_names[predicted_class_idx]

            return jsonify({
                'filename': filename,
                'predicted_class_index': predicted_class_idx,
                'predicted_label': predicted_label
            }), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return 'No image data received', 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
