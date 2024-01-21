from flask import Flask, jsonify
from flask_cors import CORS
from PIL import Image
import base64
import numpy as np
from io import BytesIO
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.generator_loader import generator_loader
from utils.visualization import generateNewImageFromAI

app = Flask(__name__)
CORS(app)  # Initialize CORS for your app

# Generate the image
noise_dimension = 100
generator = generator_loader(noise_dimension)

@app.route('/generate_image', methods=['GET'])
def api_generate_image():
    """
    API endpoint to generate and return an image using the GAN model.

    Returns:
    - flask.Response: The JSON response.
    """
    try:
        generated_image = generateNewImageFromAI(generator)

        # Convert the PIL Image to bytes
        image_bytes = BytesIO()
        generated_image.save(image_bytes, format='PNG')

        # Set the cursor position to the beginning of the BytesIO buffer
        image_bytes.seek(0)

        # Encode image to base64 for including in JSON response
        image_base64 = base64.b64encode(image_bytes.read()).decode('utf-8')

        response = {
            'success': True,
            'message': 'Image generated successfully',
            'image': image_base64,
        }

    except Exception as e:
        response = {
            'success': False,
            'message': f'Error generating image: {str(e)}',
            'image': None,
        }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
