from flask import Flask, jsonify, send_file
from PIL import Image
import numpy as np
from io import BytesIO
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.model_operations import make_or_restore_model_generator
from utils.visualization import generateNewImageFromAI

app = Flask(__name__)

@app.route('/generate_image', methods=['GET'])
def api_generate_image():
    """
    API endpoint to generate and return an image using the GAN model.

    Returns:
    - flask.Response: The image response.
    """
    # Generate the image
    noise_dimension = 100
    generator = make_or_restore_model_generator(noise_dimension)

    generated_image = generateNewImageFromAI(generator)

    # Convert the PIL Image to bytes
    image_bytes = BytesIO()
    generated_image.save(image_bytes, format='PNG')

    # Set the cursor position to the beginning of the BytesIO buffer
    image_bytes.seek(0)
    
    # Send the image bytes in the response
    return send_file(image_bytes, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)