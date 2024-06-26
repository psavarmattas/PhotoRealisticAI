# Photorealistic AI Model for Face Generation using DCGANS

[![CodeQL](https://github.com/psavarmattas/PhotoRealisticAI/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/psavarmattas/PhotoRealisticAI/actions/workflows/github-code-scanning/codeql)

## Disclaimer

This codebase is the product of research conducted for a Master's project by Puranjay Savar Mattas (@psavarmattas).

- Citation: If you use this code in your own research, please cite it using the following format:
```
Puranjay Savar Mattas (2024). Photorealistic AI Model for Face Generation using DCGANS. GitHub repository, https://github.com/psavarmattas/PhotoRealisticAI
```

- License: This code is distributed under the MIT License, which you can find in the [License](https://github.com/psavarmattas/PhotoRealisticAI/blob/main/LICENSE.MD) file.
- Trained Model: If you'd like to access the trained model for this project, you can contact Puranjay Savar Mattas at puranjaysavarmattas@gmail.com or by opening a pull request in this repository.

## Project Overview
In this project, I have defined and trained a DCGAN on a dataset of faces. My goal was to get a generator network to generate new images of faces that look as realistic as possible! At the end of the project, I was able to visualize the results of my trained Generator to see how it performs; my generated samples look like fairly realistic faces with small amounts of noise.

Due to resource constraints, we will be focusing on generating images at a lower resolution of 128x128 pixels. In this endeavor, we're utilizing the CelebFaces Attributes (CelebA) Dataset. This dataset is conveniently available on [their website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Our chosen method for image synthesis is the Deep Convolutional Generative Adversarial Networks (DCGANs). Prior to training this DCGAN model, our dataset will undergo a rigorous preprocessing phase to ensure optimal results. The ultimate objective is to train our DCGAN to synthesize new, lifelike faces that evoke the essence of the celebrities in our training set, albeit in a fabricated manner.

## Project Instructions - Training

1. Clone the repository:

    ```bash
    git clone https://github.com/psavarmattas/PhotorealisticAI.git
    ````

2. cd into the repository:
    ```bash
    cd ./PhotorealisticAI
    ```

3. Download the dataset & unzip all the images in `dataset/celebA_dataset`

4. Create a new environment with python 3.11 and activate it: 
    ```bash
    conda create --name PhotoRealisticAI python=3.11 && conda activate PhotoRealisticAI
    ```

5. Install the required packages: 
    ```bash
    pip install -r requirements.txt` or `conda install --file requirements.txt
    ```

*Note:* If you are on MacOS please add the following dependencies in your environment for compatibility `tensorflow-macos` & `tensorflow-metal`. Also add them to the the `requirements.txt` so that the program checks them and does not miss any dependencies.

6. Run the command: `python main.py`

*Note:* For any modifications or understanding of the program please go through the files as each and every function/file has it's own docstrings for detailed working.

## Project Instructions - APIs

The backend code for the project is located in the `backend` folder. Please refer to the [README](https://github.com/psavarmattas/PhotoRealisticAI/blob/main/backend/README.MD) in that folder for instructions on how to use the API.

## Project Instructions - Web Dashboard

The frontend code for the project is located in the `frontend` folder. Please refer to the [README](https://github.com/psavarmattas/PhotoRealisticAI/blob/main/frontend/README.MD) in that folder for instructions on how to start the web dashboard.

## Project Instructions - Running

1. Navigate to the root of the project

    ```bash
    cd ./PhotorealisticAI
    ```

2. Follow the [above instructions](https://github.com/psavarmattas/PhotoRealisticAI/tree/main?tab=readme-ov-file#project-instructions---training) to install the dependencies and create a new environment

3. Run the command:

    ```bash
    python run.py
    ```

4. Open your browser to `http://localhost:4200/` and generate new faces!


The frontend code for the project is located in the `frontend` folder. Please refer to the [README](https://github.com/psavarmattas/PhotoRealisticAI/blob/main/frontend/README.MD) in that folder for instructions on how to start the web dashboard.

## Project Structure

Please go through the project structure to understand the working of the project.

```css
PhotoRealisticAI/
│
├── backend/
│   ├── models/
│   │   ├── generator.py
│   ├── utils/
│   │   ├── data_generator.py
│   │   ├── generator_loader.py
│   │   └──  visualizer.py
│   ├── main.py
│   └── README.MD
│
├── ckpt/
│   ├── discriminator/
│   └── generator/
│
├── dataset/
│   └── celebA_dataset/
│
├── fig/
│
├── frontend/
│   ├── README.MD
│   ├── photorealistic-ai-frontend/
│   │   ├── node_modules/
│   │   ├── src/
│   │   │   ├── app/
│   │   │   │   ├── app.component.scss
│   │   │   │   ├── app.config.server.ts
│   │   │   │   ├── app.component.html
│   │   │   │   ├── app.component.spec.ts
│   │   │   │   ├── app.component.ts
│   │   │   │   ├── app.config.ts
│   │   │   │   └── app.routes.ts
│   │   │   └──  assets/
│   │   ├── .editorconfig
│   │   ├── .gitignore
│   │   ├── angular.json
│   │   ├── package-lock.json
│   │   ├── package.json
│   │   ├── server.ts
│   │   ├── tsconfig.app.json
│   │   ├── tsconfig.json
│   │   ├── tsconfig.spec.json
│   │   ├── index.html
│   │   ├── main.server.ts
│   │   ├── main.ts
│   │   ├── styles.scss
│   │   └── favicon.ico
│
├── models/
│   ├── discriminator.py
│   ├── gan.py
│   └── generator.py
│
├── out_metrics/
│   ├── logEpoch.txt
│   └── logFID.txt
│   ├── logMetric.txt
│   └── logVerbose.txt
│
├── plot/
│
├── utils/
│   ├── calculate_fid.py
│   ├── check_dependencies.py
│   ├── check_tf.py
│   ├── cooldown_gpu.py
│   ├── data_generation.py
│   ├── data_processing.py
│   ├── file_operations.py
│   ├── image_plotting.py
│   ├── metric_plotting.py
│   ├── model_operations.py
│   ├── multi_GPU.py
│   ├── run_servers.py
│   ├── train_load_finish.py
│   ├── train.py
│   └── visualization.py
│
├── .gitignore
├── LICENSE.MD
├── main.py
├── README.MD
├── requirements.txt
├── run.py
└── start_servers.sh
```


## [License](https://github.com/psavarmattas/PhotoRealisticAI/blob/main/LICENSE.MD)

Please make sure you read the license before using the code.

## Attribution for the dataset
The dataset is provided by `title = Deep Learning Face Attributes in the Wild author = Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou, booktitle = Proceedings of International Conference on Computer Vision (ICCV), month = December, year = 2015`
