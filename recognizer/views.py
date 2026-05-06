from django.shortcuts import render
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model(
    'recognizer/model/handwritten_model.keras'
)

def home(request):

    prediction = None

    if request.method == 'POST' and request.FILES.get('image'):

        image = request.FILES['image']

        # Open image in grayscale
        img = Image.open(image).convert('L')

        # Resize image
        img = img.resize((28, 28))

        # Convert image to numpy array
        img = np.array(img)

        # Invert colors
        img = 255 - img

        # Normalize
        img = img / 255.0

        # Reshape for CNN
        img = img.reshape(1, 28, 28, 1)

        # Predict
        pred = model.predict(img)

        # Get predicted digit
        prediction = np.argmax(pred)

    return render(request, 'index.html', {
        'prediction': prediction
    })