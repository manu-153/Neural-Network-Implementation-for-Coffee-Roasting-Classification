# Coffee Roasting Neural Network

## Overview
This repository contains a simple neural network implementation for classifying coffee roasting quality based on temperature and duration.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [Results](#results)
- [License](#license)

## Overview
This project demonstrates a simple feedforward neural network implemented using NumPy and TensorFlow for data normalization. The model predicts coffee roasting outcomes based on two input features: temperature and duration.

## Installation
To run this project, ensure you have the following dependencies installed:

```bash
pip install numpy matplotlib tensorflow
```

Additionally, ensure you have the utility functions from `lab_utils_common` and `lab_coffee_utils`.

## Usage
Run the script using:

```bash
python coffee_roasting_nn.py
```

The script:
- Loads and normalizes coffee roasting data
- Defines a simple neural network with two layers
- Makes predictions on test data
- Visualizes the decision boundary

## Code Explanation

### Data Loading and Normalization
The dataset consists of temperature and duration as input features, normalized using TensorFlow’s `Normalization` layer.

### Neural Network Implementation
The network consists of two fully connected layers, with the activation function defined as:

```python
def my_dense(a_in, W, b):
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:, j]
        z = np.dot(w, a_in) + b[j]
        a_out[j] = sigmoid(z)
    return a_out
```

A sequential model applies two layers:

```python
def my_sequential(x, W1, b1, W2, b2):
    a1 = my_dense(x, W1, b1)
    a2 = my_dense(a1, W2, b2)
    return a2
```

### Prediction and Visualization
Predictions are generated, converted into binary classification decisions, and visualized using `plt_network`.

## Results
Example predictions on test data:

```plaintext
decisions =
[[1]
 [0]]
```

The decision boundary is plotted to visualize the model’s classification regions.

## License
This project is open-source. Feel free to use and modify it.

