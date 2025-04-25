# Variational Autoencoder (VAE) Project

This repository contains a PyTorch implementation of a Variational Autoencoder (VAE) of Stable Diffusion, focusing on image generation/reconstruction, particularly with image datasets.

## Features

*   Implementation of a standard VAE architecture (Encoder, Decoder, Latent Space with Reparameterization).
*   Training script (`tools/train.py`) for model training on specified datasets.
*   Data loading utilities (`dataset/`) for datasets like Stanford Cars and custom image folders.
*   Integration with `diffusers` and `huggingface_hub` for potential model sharing or comparison.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Asura3301/vae.git
    cd vae
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    # On Windows use `venv\\Scripts\\activate`
    # On macOS/Linux use `source venv/bin/activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: PyTorch installation might vary based on your system and CUDA version. Refer to the [official PyTorch website](https://pytorch.org/get-started/locally/) for specific commands if needed.)*

## Dataset

*   **Stanford Cars:** If using `dataset/stanford_cars_loader.py` (or similar), the data might be downloaded automatically to the specified root directory (default: `./data`) or require manual download. Check the specific loader script for details.
*   **Custom Dataset:** Place your images in the `./custom/` directory following the structure below, suitable for `torchvision.datasets.ImageFolder`:
    ```
    ./custom/
        train/
            <class_name_1>/
                image1.jpg
                image2.png
                ...
            <class_name_2>/
                image3.jpeg
                ...
        test/
            <class_name_1>/
                image10.jpg
                ...
            <class_name_2>/
                image11.png
                ...
    ```
    Even if you only have one class (e.g., "dogs"), you still need the class subfolder inside `train` and `test` for `ImageFolder` to work correctly.

## Usage

Currently, hyperparameters and data paths are set directly within `tools/train.py`.

1.  **Configure:** Edit `tools/train.py` to set:
    *   Dataset paths (`train_dir`, `test_dir` in the `load_custom_image_folder` call).
    *   Hyperparameters (`num_epochs`, `batch_size`, `lr`, `beta`, `accum_steps`, `img_size`).
    *   Checkpoint directory (`checkpoint_dir`).

2.  **Run Training:**
    ```bash
    python tools/train.py
    ```
    Checkpoints will be saved in the `./checkpoints` directory (or as configured) 


## Model Architecture

The VAE implementation (`model/vae.py` or similar) consists of:

1.  **Encoder:** Maps input images to the parameters (mean $`\mu`$ and log-variance $`\log \sigma^2`$) of a distribution in the latent space. Typically implemented using convolutional layers.
2.  **Latent Space:** A lower-dimensional space where the input data is represented probabilistically.
3.  **Reparameterization Trick:** Used to sample from the latent distribution $`\mathcal{N}(\mu, \sigma^2)`$ in a way that allows backpropagation. $`z = \mu + \sigma \odot \epsilon`$, where $`\epsilon \sim \mathcal{N}(0, I)`$.
4.  **Decoder:** Maps points sampled from the latent space back to the original image space. Typically implemented using transposed convolutional layers ( deconvolutions).

The model is trained by minimizing a loss function composed of:
*   **Reconstruction Loss:** Measures how well the decoder reconstructs the input image (e.g., Mean Squared Error or Binary Cross-Entropy).
*   **KL Divergence Loss:** A regularization term that encourages the learned latent distribution to be close to a standard normal distribution $`\mathcal{N}(0, I)`$. $`D_{KL}(q(z|x) || p(z))`$

*(Specific layer configurations and hyperparameters can be found in the model definition file.)*

## License

MIT