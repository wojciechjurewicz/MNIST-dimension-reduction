# MNIST Dimension Reduction & Analysis

**Author: Wojciech Jurewicz**

A project for the "Machine Learning" course at the Technical University of Lodz.

## Project Overview
This project explores dimension reduction techniques on the MNIST dataset using PyTorch Autoencoders. It compares the performance of a Random Forest classifier on raw data versus data compressed into a latent space. Additionally, it investigates dataset noise ("dusty" images) and visualizes the reconstruction capabilities of Dense and Convolutional Autoencoders.

## Key Features
- **Data Preprocessing**: Normalization and reshaping for both classical ML (flattened) and Deep Learning (channel-first) models.
- **Baseline Model**: Random Forest classifier trained on raw 784-dimensional data.
- **Architecture Search**: Comparison of Dense and Convolutional Autoencoders across various latent dimensions (4 to 64) to find the optimal balance between compression and reconstruction loss.
- **Noise Analysis**: Custom logic to detect and visualize noisy "dusty" images in the test set.
- **Final Evaluation**: Benchmarking compression ratio, training time, and classification accuracy retention.

## Project Structure
- `data/`: Contains raw MNIST binary files and processed `.npz` data.
- `models/`: Stores trained PyTorch models (`.pth`).
- `results/`: Stores metrics and baseline results.
- `preprocessing_and_baseline_model.ipynb`: Loads data, preprocesses it, and establishes a Random Forest baseline.
- `architecture.ipynb`: Defines Autoencoder architectures, performs hyperparameter search for latent dimension, and trains final models.
- `noise_investigaiton_and_latent_analysis.ipynb`: Loads trained models, detects noisy images, and visualizes reconstructions.
- `final_evaluation.ipynb`: Compresses the dataset using the best encoder, trains a classifier on latent vectors, and compares results with the baseline.

## Usage
Run the notebooks in the following order to reproduce the results:
1.  **`preprocessing_and_baseline_model.ipynb`**: Prepare data and run baseline.
2.  **`architecture.ipynb`**: Train Autoencoders.
3.  **`noise_investigaiton_and_latent_analysis.ipynb`**: Analyze results and visualize.
4.  **`final_evaluation.ipynb`**: Run final benchmarks.

## Results Summary
- **Compression**: The Convolutional Autoencoder with a latent dimension of 32 achieved a **24.5x compression ratio** (784 -> 32 dimensions).
- **Accuracy**: The Random Forest classifier maintained high accuracy on the compressed data (**95.60%**) compared to the baseline on raw data (**96.92%**).
- **Efficiency**: Training on compressed data was more efficient. It took 2.23s instead of 3.34s for the baseline model.
- **Noise Removal**: Autoencoders successfully removed noise from "dusty" images, effectively acting as denoisers and reconstructing clean digits from corrupted inputs.