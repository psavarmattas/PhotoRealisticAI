# Backend API README

`Please note this project is still under development and the README & the code are still being updated.`

This folder contains the backend code for the Photorealistic AI project. The main purpose of the backend is to provide an API endpoint (`/generate_image`) to generate and return images using the trained GAN model.

### Overview
The `generate_images_api.py` script defines a Flask web application with a single API endpoint. This endpoint is responsible for generating and returning an image using the trained GAN model.

### How to Use

1. Clone the repository:

    ```bash
    git clone https://github.com/psavarmattas/PhotorealisticAI.git
    ```

2. Navigate to the project root:

    ```bash
    cd ./PhotorealisticAI
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the Flask application:

    ```bash
    python backend/generate_images_api.py
    ```

5. Once the application is running, you can make a GET request to the following endpoint to generate and receive an image:

    ```bash
    http://127.0.0.1:5000/generate_image
    ```

    Example using cURL:

    ```bash
    curl http://127.0.0.1:5000/generate_image --output generated_image.png
    ```

 The Flask app will be accessible at `http://127.0.0.1:5000`. To generate an image, make a GET request to `http://127.0.0.1:5000/generate_image`.

### API Endpoint

- **Endpoint**: `/generate_image`
- **Method**: GET
- **Response**: Returns the generated image in PNG format.

## Note
Ensure that the GAN model is trained and saved before using the API. Refer to the main README for training instructions.

For MacOS users, please check the dependencies mentioned in the main README for compatibility.

## [License](https://github.com/psavarmattas/PhotoRealisticAI/blob/main/LICENSE.MD)

Please make sure you read the license before using the code.

## Attribution for the dataset
The dataset is provided by `title = Deep Learning Face Attributes in the Wild author = Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou, booktitle = Proceedings of International Conference on Computer Vision (ICCV), month = December, year = 2015`