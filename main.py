from Classes.Layer import Layer_Dense
from Classes.Activation_Functions import ReLU
from Classes.Loss_Functions import CrossEntropyLoss
from Classes.Neural_Network import NeuralNetwork
import numpy as np
import pandas as pd

# Load training data
train_csv = 'mnist-dataset/mnist_train.csv'
train_data = pd.read_csv(train_csv)
train_labels = train_data.iloc[:, 0].values
train_pixels = train_data.iloc[:, 1:].values / 255.0

# Load testing data
test_csv = 'mnist-dataset/mnist_test.csv'
test_data = pd.read_csv(test_csv)
test_labels = test_data.iloc[:, 0].values
test_pixels = test_data.iloc[:, 1:].values / 255.0

# Network dimensions
input_dim = 784
hidden_dim1 = 256
hidden_dim2 = 128
output_dim = 10

# Initialize neural network
Digit_Recognizer = NeuralNetwork(CrossEntropyLoss())
Digit_Recognizer.add_layer(Layer_Dense(input_dim, hidden_dim1, ReLU()))
Digit_Recognizer.add_layer(Layer_Dense(hidden_dim1, hidden_dim2, ReLU()))
Digit_Recognizer.add_layer(Layer_Dense(hidden_dim2, output_dim, None))

# Training parameters
epochs = 20
batch_size = 32
learning_rate = 0.05

epoch_losses = []

# Accuracy function
def accuracy(predictions, labels):
    predicted_classes = np.argmax(predictions, axis=1)
    return np.mean(predicted_classes == labels)

# Training loop
for epoch in range(epochs):
    indices = np.arange(train_pixels.shape[0])
    np.random.shuffle(indices)
    train_pixels = train_pixels[indices]
    train_labels = train_labels[indices]

    batch_losses = []
    for batch_start in range(0, train_pixels.shape[0], batch_size):
        batch_end = batch_start + batch_size
        inputs_batch = train_pixels[batch_start:batch_end]
        labels_batch = train_labels[batch_start:batch_end]

        # Forward pass
        predictions = Digit_Recognizer.forward(inputs_batch)
        loss = Digit_Recognizer.loss.forward(predictions, labels_batch)
        batch_losses.append(loss)
        
        # Backward pass and parameter update
        Digit_Recognizer.backward(labels_batch)
        Digit_Recognizer.update(learning_rate)

        if (batch_start // batch_size + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}, Batch {batch_start // batch_size + 1}, Loss: {loss:.4f}')
    
    average_loss = np.mean(batch_losses)
    epoch_losses.append(average_loss)
    print(f'Epoch {epoch + 1}, Average Loss: {average_loss:.4f}')

# Test the network
test_predictions = Digit_Recognizer.forward(test_pixels)
test_accuracy = accuracy(test_predictions, test_labels)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
