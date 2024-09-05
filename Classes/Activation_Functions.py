import numpy as np

class ReLU:
    # Forward pass: Applies ReLU activation function
    def forward(self, z):
        self.cache = z  # Store 'z' for use in the backward pass
        return np.maximum(0, z)  # Return max between 0 and 'z', element-wise
    
    # Backward pass: Compute gradient of ReLU function with respect to 'z'
    def backward(self, dA, z):
        dZ = dA * (z > 0)  # Derivative of ReLU is 1 for positive 'z' and 0 otherwise
        return dZ  # Return the gradient to propagate it backward

class Sigmoid:
    # Forward pass: Applies the sigmoid activation function
    def forward(self, z):
        self.cache = z  # Store 'z' for use in the backward pass
        return 1 / (1 + np.exp(-z))  # Sigmoid function
    
    # Backward pass: Compute the gradient of the sigmoid function with respect to 'z'
    def backward(self, dA, z):
        sigmoid = 1 / (1 + np.exp(-z))  # Calculate sigmoid(z) again
        dZ = dA * sigmoid * (1 - sigmoid)  # Derivative of sigmoid
        return dZ  # Return the gradient to propagate it backward


class Softmax:
    def forward(self, z):
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        self.softmax_probs = e_z / np.sum(e_z, axis=1, keepdims=True)
        return self.softmax_probs
    
    def backward(self, dA, y_true):
        batch_size = y_true.shape[0]
        dZ = self.softmax_probs.copy()
        y_true_indices = np.argmax(y_true, axis=1)
        dZ[np.arange(batch_size), y_true_indices] -= 1
        dZ /= batch_size
        return dZ
