# MNIST Handwritten Digit Classifier in C

This project implements a single-hidden-layer neural network from scratch in C to classify handwritten digits from the MNIST dataset. It avoids high-level libraries (like TensorFlow or PyTorch), relying solely on standard libraries and raw mathematical implementations of backpropagation.

## Project Overview

The classifier is designed to read 28x28 pixel images (flattened to 784 inputs) and predict the digit (0-9). By using a hidden layer of 128 neurons, the model can achieve over 95% accuracy on unseen test data.

## Features & Implementation Details

### 1. Neural Network Architecture
- **Input Layer:** 784 neurons (representing the 28x28 pixels of an MNIST image).
- **Hidden Layer:** 128 neurons utilizing the **ReLU (Rectified Linear Unit)** activation function for non-linearity.
- **Output Layer:** 10 neurons (one for each digit 0-9) utilizing the **Softmax** activation function to produce a probability distribution.

### 2. Training Mechanics
- **Backpropagation:** Custom implementation of gradient descent to update weights and biases based on the error.
- **Loss Function:** Negative Log-Likelihood (NLL) calculated from the Softmax output.
- **Weight Initialization:** The code is designed to load initial weights and biases from external files (`init_guesshlayer.txt` and `init_guessolayer.txt`) to ensure reproducible starts.
- **Shuffling:** Includes a shuffling mechanism to reorder training data every epoch, preventing the model from learning the sequence of the data rather than the actual patterns.
- **L2 Regularization:** Implements weight decay (lambda = 0.001) in the output layer updates to prevent overfitting and encourage generalization.

### 3. Memory Management
- Utilizes dynamic memory allocation (`malloc`) to handle large datasets (e.g., 10,000 training images) on the heap rather than the stack.
- Includes thorough memory cleanup (`free`) to prevent leaks after training and inference.

### 4. Data Handling
- **Training Data:** Reads from `inputs.txt` containing 10,000 normalized samples.
- **Test Data:** Reads from `test_inputs.txt` for inference on 1,000 unseen samples.
- **Output Logging:** Generates `test_results.txt` to provide a detailed CSV-style log of every prediction versus the actual ground truth.

## Code Structure

- `Layer` Struct: Defines the properties of each neural layer, including weights, biases, outputs, and deltas.
- `Relu()`: Implements the activation function $f(x) = \max(0, x)$.
- `softmax()`: Converts raw output scores into probabilities.
- `predict()`: Performs a forward pass to classify a single input image.
- `shuffle()`: Randomizes the training data order.
- `main()`: Orchestrates the loading of data, the training loop (500 cycles), and the final inference tests.

## Mathematical Summary

The core weight update logic follows:
$$w_{new} = w_{old} - \eta \cdot (rac{\partial Loss}{\partial w} + \lambda \cdot w)$$
Where:
- $\eta$ (eta) is the learning rate (set to 0.001).
- $\lambda$ (lambda) is the regularization strength.

## Performance Benchmark
In a typical run with 10,000 training images and a learning rate of 0.001:
- **Training Accuracy:** Often reaches near 98-100%.
- **Test Accuracy:** Approximately **95.3%** on unseen data.
