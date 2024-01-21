from flask import Flask, jsonify, send_file
from PIL import Image
import numpy as np
from io import BytesIO
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.generator_loader import generator_loader
from utils.visualization import generateNewImageFromAI

app = Flask(__name__)

# Generate the image
noise_dimension = 100
generator = generator_loader(noise_dimension)

@app.route('/generate_image', methods=['GET'])
def api_generate_image():
    """
    API endpoint to generate and return an image using the GAN model.

    Returns:
    - flask.Response: The image response.
    """

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