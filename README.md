# Custom Neural Network Implementation in C++

A lightweight, dependency-free neural network implementation built from scratch in C++ only using STL .

## Overview

This project implements a fully functional feedforward neural network with backpropagation, built entirely from the ground up using only standard C++ libraries. The implementation demonstrates core deep learning concepts including forward propagation, backpropagation, and gradient descent optimization.

## Features

- **Pure C++ Implementation** - No external ML libraries required (only standard C++ headers)
- **Flexible Architecture** - Configurable number of hidden layers and neurons per layer
- **ReLU Activation** - Uses ReLU (Rectified Linear Unit) activation function
- **Backpropagation** - Full implementation of gradient descent with backpropagation
- **Regression Support** - Designed for regression tasks with MSE loss
- **Random Weight Initialization** - Xavier-style initialization for weights
- **Interactive Training** - User-friendly CLI for configuring and training the network

## Architecture

The neural network consists of:
- **Input Layer** - Automatically sized based on feature dimensions
- **Hidden Layers** - User-configurable (number and size)
- **Output Layer** - Single neuron for regression tasks
- **Activation Function** - ReLU for hidden layers
- **Loss Function** - Mean Squared Error (MSE)
- **Optimizer** - Vanilla Gradient Descent

## Getting Started

### Prerequisites

- C++ compiler with C++11 support or higher (g++, clang++, etc.)
- No additional libraries required


## Usage

When you run the program, you'll be prompted to configure the network:

1. **Number of Hidden Layers** - Enter how many hidden layers you want
2. **Neurons per Layer** - Specify the number of neurons for each hidden layer
3. **Number of Epochs** - Define how many training iterations to run


## Dataset

The implementation includes the Boston Housing dataset (subset) as a demonstration, with 44 training samples. Each sample contains 13 features predicting house prices.

Features include:
- Crime rate
- Residential land zoning
- Non-retail business acres
- Charles River proximity
- NOx concentration
- Average rooms per dwelling
- And more...

## Code Structure

### Main Components

**Neuron Class**
- Stores weights and bias
- Performs forward pass computation
- Applies activation function

**Network Creation Functions**
- `inputLayer()` - Creates input neurons
- `hiddenLayer()` - Creates hidden layer neurons
- `regressionOutputLayer()` - Creates output neuron

**Training Functions**
- `forwardProp()` - Forward propagation through the network
- `backProp()` - Backpropagation and weight updates
- `train()` - Main training loop with MAE tracking

### Key Parameters

- **Learning Rate**: 0.001
- **Weight Initialization Range**: [-0.15, 0.15]
- **Initial Bias**: 0.01
- **Activation**: ReLU (max(0, x))

## Performance Metrics

The network reports Mean Absolute Error (MAE) after each epoch, allowing you to monitor training progress.

## Customization

You can easily modify the code to:
- Add different activation functions
- Implement classification (modify output layer)
- Add regularization techniques
- Implement different optimizers (Adam, RMSprop, etc.)
- Use different loss functions
- Add dropout or batch normalization

## Limitations

- Currently supports only regression tasks
- Vanilla gradient descent (no advanced optimizers)
- No validation set splitting
- ReLU activation (no other activations)

## Future Enhancements

- [ ] Add support for classification tasks
- [ ] Implement additional activation functions (Sigmoid, Tanh, Leaky ReLU)
- [ ] Add cross-validation
- [ ] Implement mini-batch gradient descent
- [ ] Add learning rate scheduling
- [ ] Support for saving/loading trained models
- [ ] Visualization of training metrics

## Contributing

Feel free to fork this repository and submit pull requests with improvements or bug fixes.

## Author

**Brainard Philemon Jagati**

## License

This project is open source and available for educational purposes.

## Acknowledgments

Built as a learning exercise to understand neural networks from first principles without relying on high-level frameworks.

---

*"The best way to understand neural networks is to build one from scratch!"*
