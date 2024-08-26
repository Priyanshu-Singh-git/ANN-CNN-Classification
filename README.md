# MNIST Classification using ANN and CNN

This project demonstrates image classification on the MNIST dataset using two different neural network architectures: Artificial Neural Network (ANN) and Convolutional Neural Network (CNN). The models are implemented using PyTorch and are trained to classify handwritten digits from the MNIST dataset.

## Project Overview

1. **Data Preparation**
   - The MNIST dataset is loaded and preprocessed. Images are resized to 64x64 pixels, normalized, and split into training and testing sets.
   - The dataset is loaded using PyTorch's `DataLoader` for efficient batching and shuffling.

2. **Models**
   - **Artificial Neural Network (ANN):**
     - A fully connected network with multiple hidden layers and dropout for regularization.
     - Model is defined in the `Neural_net` class.
   - **Convolutional Neural Network (CNN):**
     - A deep network with multiple convolutional layers, pooling layers, batch normalization, and dropout.
     - Model is defined in the `CNN_net` class.

3. **Training**
   - Both models are trained using Cross-Entropy Loss and the AdamW optimizer.
   - **Mixed-Precision Training:**
     - **Mixed-precision training** is employed using PyTorch's automatic mixed precision (AMP) to enhance training efficiency and performance.
     - AMP uses both 16-bit and 32-bit floating-point numbers during training to reduce memory usage and increase computational speed while maintaining numerical stability.
     - The `torch.cuda.amp` module provides `GradScaler` and `autocast` functions to facilitate mixed-precision training.
     - **GradScaler**: Scales the loss to prevent gradients from becoming too small and adjusts them accordingly during backpropagation.
     - **autocast**: Automatically chooses the appropriate precision for operations to balance performance and precision.
   - Learning rate scheduling and mixed-precision training are employed to improve training efficiency and performance.

4. **Evaluation**
   - **ANN Model:** Achieved a maximum accuracy of approximately 89% on the MNIST test set.
   - **CNN Model:** Achieved a maximum accuracy of approximately 98.6% on the MNIST test set.
   - Training progress is monitored by plotting loss and accuracy curves.
   - The trained models are saved as `.pth` files for future use.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib




