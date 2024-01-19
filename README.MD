# Photorealistic AI Model for Face Generation using DCGANS

`Please note this project is still under development and the README & the code are still being updated.`

## Project Overview
In this project, I have defined and trained a DCGAN on a dataset of faces. My goal was to get a generator network to generate new images of faces that look as realistic as possible! At the end of the project, I was able to visualize the results of my trained Generator to see how it performs; my generated samples look like fairly realistic faces with small amounts of noise.

Due to resource constraints, we will be focusing on generating images at a lower resolution of 128x128 pixels. In this endeavor, we're utilizing a subset of the CelebFaces Attributes (CelebA) Dataset, specifically the first 50,000 images. This subset is conveniently available on Kaggle. Our chosen method for image synthesis is the Deep Convolutional Generative Adversarial Networks (DCGANs). Prior to training this DCGAN model, our dataset will undergo a rigorous preprocessing phase to ensure optimal results. The ultimate objective is to train our DCGAN to synthesize new, lifelike faces that evoke the essence of the celebrities in our training set, albeit in a fabricated manner.
    
## Project Instructions
1. Clone the repository:
`git clone https://github.com/psavarmattas/PhotorealisticAI.git`

2. cd into the repository: `cd ./PhotorealisticAI`

3. Create a new environment with python 3.11 and activate it: `conda create --name PhotoRealisticAI python=3.11 && conda activate PhotoRealisticAI`

4. Install the required packages: `pip install -r requirements.txt` or `conda install --file requirements.txt`

<I><B>Note:</B> The above requirements contain a few MacOS specific requirements please make sure not to install those if you are not on MacOS & remove them from the `requirements.txt` as without removing it from the file the main file will not run.</I>


5. Run the `main.py`!

## Steps of the algorithm's working

### Step 1 | Import Necessary Libraries

First of all, let's import necessary libraries:

### Step 2 | Loading and Preprocessing the Dataset

The first step in our pipeline involves preparing our data for the DCGAN model. For this project, we're using  the CelebFaces Attributes (CelebA) Dataset, which is available on [their website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

To ensure our model receives optimal data, we perform the following preprocessing steps on each image:

1. Read Image: We start by obtaining the path to every image within the dataset directory.

2. Crop: Each image gets cropped, removing 20 pixels from the top and bottom, resulting in a square image.

3. Resize: Given the computational constraints and our DCGAN's architecture, we resize the images to a resolution of 128x128 pixels.

4. Normalize: Images are converted to numpy arrays and then normalized to the range [-1, 1] to facilitate the neural network's training process.

By the end of this preprocessing phase, our dataset will be a numpy array with the shape indicating the number of images and the dimensions of each image.

Having preprocessed our dataset, it's now time to visually examine the images that we'll be using to train our DCGAN model. Here, we'll displaying a small sample of the processed images.

<I><B>Note:</B> Given constraints on computational resources and the desire to streamline training times, we have opted to train our model on a subset of 20,000 images. However, if resources permit, training with a larger dataset can potentially enhance the model's performance and yield improved results.</I>

### Step 3 | Define DCGAN Architecture

In this part, we define the architecture for the Discriminator, Generator, and the combined GAN model:

1. The Discriminator is a Convolutional Neural Network (CNN) that classifies whether an image is real or fake (generated).

2. The Generator is a CNN that upsamples a random noise vector into an image (fake image).

3. The combined GAN model, where the Discriminator's trainability is frozen, is used for training the Generator. It takes noise as input and the Discriminator classifies the generated images. In this setup, the Generator learns to create images that the Discriminator is unable to distinguish from real images.

### Step 4 | Prepare Data for Training DCGAN

Afterward, I am going to define some functions help in creating training batches for the Discriminator and Generator:

1. generate_real_samples - Selects random real images from the dataset and labels them as 1, indicating they're real.

2. generate_noise_samples - Produces random noise vectors. These are the inputs to the Generator, which it uses to create fake images.

3. generate_fake_samples - Uses the Generator to produce fake images from the noise vectors and labels them as 0, indicating they're fake.

Together, these functions ensure that the Discriminator is trained to distinguish between real and fake images, while the Generator tries to produce images that the Discriminator can't differentiate from real ones.

### Step 5 | Train DCGAN & Monitor Progress

Next, I am going to handle the simultaneous training of the Generator and Discriminator in our GAN model:

1. generate_images - This function creates images from noise using the current state of the Generator for a given epoch. The images created during training are saved and can be visualized later to provide a glimpse into the Generator's evolving performance.

2. display_saved_images - After training concludes, this function visualizes the saved images, presenting the progression of the Generator's capability across the epochs.

3. plot_generated_images - This function generates and plots images from noise using the current state of the Generator. It is utilized to visually track the Generator's performance during epochs as indicated in the training output verbose.

4. train - This function orchestrates the core GAN training process. It runs through all epochs, and within each epoch, iterates over all batches. For every batch:

    1. It first generates a batch of real images and a batch of fake images (created by the Generator), tagging each set with the appropriate labels. The Discriminator is then trained on these batches, and both its loss and accuracy metrics are computed.

    2. Subsequently, a batch of noise samples is created, with the aim of deceiving the Discriminator into accepting these as genuine. As the Generator strives to amplify the loss, it learns to produce increasingly authentic-looking images.

The above process is iteratively executed until all epochs are finalized. The plot_generated_images function is used to visually track the Generator's progress in real-time during training, whereas the display_saved_images function is used at the end of training to showcase the stored images, shedding light on the Generator's refinement over the training epochs.

This combined training approach allows the Generator and Discriminator to learn together. The Discriminator improves at distinguishing fake images, and the Generator becomes better at fooling the Discriminator.

<I><B>Note:</B> A higher number of epochs could potentially lead to even more realistic and refined results.</I>