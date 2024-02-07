import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from skimage import io
from tensorflow.keras.preprocessing import image


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()

model =tf.keras.models.load_model('PlantDNet.h5',compile=False)

disease_solutions = {
    'Pepper__bell___Bacterial_spot': 'Apply copper-based fungicides and practice crop rotation.',
    'Pepper__bell___healthy': 'No action required. Maintain plant health and monitor for pests.',
    'Potato___Early_blight': 'Remove infected leaves and apply fungicides containing chlorothalonil.',
    'Potato___Late_blight': 'Remove and destroy infected plants. Apply fungicides containing chlorothalonil or mancozeb.',
    'Potato___healthy': 'Ensure proper irrigation and fertilization. Practice crop rotation.',
    'Tomato_Bacterial_spot': 'Remove infected leaves and apply copper-based fungicides. Practice crop rotation.',
    'Tomato_Early_blight': 'Remove infected leaves and apply fungicides containing chlorothalonil.',
    'Tomato_Late_blight': 'Remove and destroy infected plants. Apply fungicides containing chlorothalonil or mancozeb.',
    'Tomato_Leaf_Mold': 'Ensure good air circulation and avoid overhead watering. Apply fungicides containing chlorothalonil or mancozeb.',
    'Tomato_Septoria_leaf_spot': 'Remove infected leaves and apply fungicides containing chlorothalonil or copper-based fungicides.',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Apply insecticidal soap or neem oil to control spider mites. Prune affected leaves.',
    'Tomato__Target_Spot': 'Remove infected leaves and apply fungicides containing chlorothalonil or copper-based fungicides.',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Remove and destroy infected plants. Control whiteflies and other vectors.',
    'Tomato__Tomato_mosaic_virus': 'Remove and destroy infected plants. Control aphids and other vectors.',
    'Tomato_healthy': 'No action required. Maintain plant health and monitor for pests.'
}


def model_predict(img_path, model):
    img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    show_img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.array(x, 'float32')
    x /= 255
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html') #render home page

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        print(preds[0])

        disease_class = [
            'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
            'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
            'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
            'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
            'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
        ]
        prediction_index = np.argmax(preds)
        predicted_disease = disease_class[prediction_index]
        
        # Get solution for predicted disease
        solution = disease_solutions.get(predicted_disease, "Solution not found.")

        return {
            "predicted_disease": predicted_disease,
            "solution": solution
        }


if __name__ == '__main__':
    app.run(debug=True)
