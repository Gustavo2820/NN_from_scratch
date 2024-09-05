# NN_from_scratch
This repository contains an implementation of a neural network built from scratch using Python. The project focuses on creating a feedforward neural network to classify handwritten digits from the MNIST dataset.

## Features
- Feedforward neural network built from scratch without using high-level libraries like TensorFlow or PyTorch.
- Trains on the MNIST dataset (handwritten digits).
- Supports multiple hidden layers and activation functions.

## Requirements
Before running the project, ensure you have the following dependencies installed: numpy

Install using:
```bash
pip install numpy
```

Additionally, you will need to manually download the MNIST dataset since it is too large to be uploaded to this repository.

1. **Download the MNIST dataset:**
   - Visit [this Kaggle page](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) and download the `mnist_train.csv` and `mnist_test.csv` files.
   - After downloading, unzip and place the files in a folder named `mnist-dataset/` within the project directory.

## Getting Started

1. **Clone the repository:**
```bash
   git clone https://github.com/Gustavo2820/NN_from_scratch.git
   cd NN_from_scratch
```

2. **Run:**
```bash
   python main.py
```

3. **Configuration:**
   You can configure the following parameters in the `main.py` file:
   - Learning rate
   - Number of epochs
   - Batch size
   - Network architecture (number of hidden layers and neurons per layer)

## How It Works

1. **Feedforward Process:**
   - The input layer receives the image (28x28 pixels) flattened into a 784-dimensional vector.
   - The data is passed through hidden layers with activation functions like ReLU or Sigmoid.
   - The final output layer consists of 10 neurons, each representing a digit from 0 to 9, with softmax used to compute probabilities.

2. **Backpropagation:**
   - The network uses backpropagation to calculate gradients for weights and biases.
   - The `NeuralNetwork` class manages the forward pass, loss calculation, and backward pass to adjust the parameters.

3. **Loss Function:**
   - The cross-entropy loss function is used to measure the model's performance during training.

4. **Optimization:**
   - Stochastic Gradient Descent (SGD) is used to minimize the loss function, with the learning rate controlling the step size.

## Results
Once the network is trained, it achieves an accuracy of approximately 98% on the MNIST test set after 20 epochs.
