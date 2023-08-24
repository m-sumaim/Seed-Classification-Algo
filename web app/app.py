from flask import Flask, request, render_template
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('/home/sumaim/Documents/web app/corn_seed_classifier.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image
        img = request.files['image']

        # Read the image file into memory
        img = BytesIO(img.read())

        # Load the image and resize it to the required size
        img = image.load_img(img, target_size=(150, 150))

        # Convert the image to a numpy array and normalize it
        x = image.img_to_array(img)
        x = x / 255.0

        # Add a new axis to create a batch of size 1
        x = np.expand_dims(x, axis=0)

        # Make a prediction on the image
        predictions = model.predict(x)

        # Get the index of the class with the highest probability
        class_index = np.argmax(predictions[0])

        # Get the name of the class from the class index
        class_names = ['Broken', 'Discolored', 'Pure', 'Silkcut']
        predicted_class_name = class_names[class_index]

        # Get the probability of the predicted class
        class_probability = predictions[0][class_index]

        return render_template('index.html', predicted_class_name=predicted_class_name, class_probability=class_probability)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
