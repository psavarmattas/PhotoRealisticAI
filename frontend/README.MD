# PhotoRealistic AI Frontend

`Please note this project is still under development and the README & the code are still being updated.`

Welcome to the frontend of the PhotoRealistic AI project! This Angular application integrates with a backend API to generate and display photorealistic images using advanced AI algorithms.

## Getting Started

Follow these instructions to set up and run the frontend locally.

### Prerequisites

Make sure you have the following installed:

- Node.js and npm
- Angular CLI

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/psavarmattas/PhotorealisticAI.git
   ``````

2. Navigate to the project root:

    ```bash
    cd ./PhotorealisticAI
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Install node:
    ```bash
    npm install
    ```

### Running the Application

1. Run the Backend API:

    ```bash
    cd backend
    python generate_images_api.py
    ```

**Note:** Make sure you see the backend API running at http://localhost:5000/generate_image as it wil output the generated image.

2. Run the development server:

    ```bash
    cd frontend/photorealistic-ai-frontend
    ng serve
    ```

**Note:** Open your browser and navigate to http://localhost:4200/ to view the app.

### Usage

- Explore the AI-generated images.
- Click the `Load Image` button to fetch a new image from the backend API.

## Note
Ensure that the GAN model is trained and saved before using the API. Refer to the main README for training instructions.

For MacOS users, please check the dependencies mentioned in the main README for compatibility.

## [License](https://github.com/psavarmattas/PhotoRealisticAI/blob/main/LICENSE.MD)

Please make sure you read the license before using the code.

## Attribution for the dataset
The dataset is provided by `title = Deep Learning Face Attributes in the Wild author = Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou, booktitle = Proceedings of International Conference on Computer Vision (ICCV), month = December, year = 2015`