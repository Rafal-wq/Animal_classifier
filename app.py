from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

model = tf.keras.models.load_model('animal_model.h5')

with open('class_names.txt') as f:
    class_names = [line.strip() for line in f.readlines()]

IMG_SIZE = (224, 224)

@app.route('/')
def index():
    return render_template('index.html', class_names=class_names)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Brak pliku'})

    file = request.files['file']
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = tf.expand_dims(np.array(img), 0)

    predictions = model.predict(img_array)
    scores = predictions[0].tolist()

    results = [
        {'animal': class_names[i], 'probability': round(scores[i] * 100, 1)}
        for i in range(len(class_names))
    ]
    results.sort(key=lambda x: x['probability'], reverse=True)

    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True)
