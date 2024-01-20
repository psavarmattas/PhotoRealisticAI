# Photorealistic AI Model for Face Generation using DCGANS

`Please note this project is still under development and the README & the code are still being updated.`

## Project Overview
In this project, I have defined and trained a DCGAN on a dataset of faces. My goal was to get a generator network to generate new images of faces that look as realistic as possible! At the end of the project, I was able to visualize the results of my trained Generator to see how it performs; my generated samples look like fairly realistic faces with small amounts of noise.

Due to resource constraints, we will be focusing on generating images at a lower resolution of 128x128 pixels. In this endeavor, we're utilizing the CelebFaces Attributes (CelebA) Dataset. This dataset is conveniently available on [their website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Our chosen method for image synthesis is the Deep Convolutional Generative Adversarial Networks (DCGANs). Prior to training this DCGAN model, our dataset will undergo a rigorous preprocessing phase to ensure optimal results. The ultimate objective is to train our DCGAN to synthesize new, lifelike faces that evoke the essence of the celebrities in our training set, albeit in a fabricated manner.
    
## Project Instructions
1. Clone the repository:
`git clone https://github.com/psavarmattas/PhotorealisticAI.git`

2. cd into the repository: `cd ./PhotorealisticAI`

3. Download the dataset & unzip all the images in `dataset/celebA_dataset`

4. Create a new environment with python 3.11 and activate it: `conda create --name PhotoRealisticAI python=3.11 && conda activate PhotoRealisticAI`

5. Install the required packages: `pip install -r requirements.txt` or `conda install --file requirements.txt`

<I><B>Note:</B> If you are on MacOS please add the following dependencies in your environment for compatibility `tensorflow-macos` & `tensorflow-metal`. Also add them to the the `requirements.txt` so that the program checks them and does not miss any dependencies.</I>

6. Run the command: `python main.py`

<I><B>Note:</B> For any modifications or understanding of the program please go through the files as each and every function/file has it's own docstrings for detailed working</I>

## [License](https://github.com/psavarmattas/PhotoRealisticAI/blob/main/LICENSE.MD)

Please make sure you read the license before using the code.

## Attribution for the dataset
The dataset is provided by `title = Deep Learning Face Attributes in the Wild author = Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou, booktitle = Proceedings of International Conference on Computer Vision (ICCV), month = December, year = 2015`