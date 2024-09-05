import numpy as np

class Layer_Dense:
    def __init__(self, input_dim, output_dim, activation=None):
        self.weights = np.random.randn(input_dim, output_dim) * 0.01  # Initialize weights
        self.biases = np.zeros((1, output_dim))  # Initialize biases
        self.activation = activation  # Activation function (if any)

    def forward(self, inputs):
        self.inputs = inputs  # Store inputs
        self.z = np.dot(inputs, self.weights) + self.biases  # Compute weighted sum plus biases
        if self.activation:  # Apply activation function if provided
            self.a = self.activation.forward(self.z)
        else:
            self.a = self.z  # No activation
        return self.a
    
    def backward(self, dA):
        if self.activation:
            dZ = self.activation.backward(dA, self.z)
        else:
            dZ = dA
    
        self.dW = np.dot(self.inputs.T, dZ)  # Gradient w.r.t. weights
        self.dB = np.sum(dZ, axis=0, keepdims=True)  # Gradient w.r.t. biases
        dA_prev = np.dot(dZ, self.weights.T)  # Gradient w.r.t. previous layer inputs
        return dA_prev

    def update_params(self, learning_rate):
        self.weights -= learning_rate * self.dW
        self.biases -= learning_rate * self.dB
