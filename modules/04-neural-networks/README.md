# Module 4: Neural Networks

## Overview
Feedforward neural networks, backpropagation, and deep learning fundamentals.

## Key Concepts

### 1. Neural Network Architecture

**Single neuron** (perceptron):
```
z = w^T x + b
a = σ(z)
```

**Multi-layer network**:
```
Input → Hidden Layer 1 → Hidden Layer 2 → Output

Layer l: z^(l) = W^(l) a^(l-1) + b^(l)
         a^(l) = σ(z^(l))
```

### 2. From Scratch Implementation

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        layer_sizes: [input_dim, hidden1_dim, hidden2_dim, ..., output_dim]
        Example: [784, 128, 64, 10] for MNIST
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        # Initialize weights (Xavier initialization)
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, z):
        """ReLU activation"""
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        """Softmax for output layer"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        """
        Forward pass

        Returns:
            activations: list of activations for each layer
            z_values: list of pre-activation values
        """
        activations = [X]
        z_values = []

        for i in range(self.num_layers - 1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            z_values.append(z)

            # Activation
            if i < self.num_layers - 2:
                # Hidden layers: ReLU
                a = self.relu(z)
            else:
                # Output layer: Softmax
                a = self.softmax(z)

            activations.append(a)

        return activations, z_values

    def backward(self, X, y, activations, z_values, learning_rate):
        """
        Backpropagation

        Compute gradients and update weights
        """
        m = X.shape[0]

        # Output layer gradient
        dz = activations[-1] - y  # For softmax + cross-entropy

        # Backpropagate through layers
        for i in reversed(range(self.num_layers - 1)):
            # Gradients
            dw = activations[i].T @ dz / m
            db = np.sum(dz, axis=0, keepdims=True) / m

            # Update parameters
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db

            # Gradient for previous layer
            if i > 0:
                dz = (dz @ self.weights[i].T) * self.relu_derivative(z_values[i-1])

    def train(self, X, y, epochs=100, learning_rate=0.01, batch_size=32):
        """Training loop"""
        n = X.shape[0]

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Mini-batch gradient descent
            for i in range(0, n, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                # Forward pass
                activations, z_values = self.forward(X_batch)

                # Backward pass
                self.backward(X_batch, y_batch, activations, z_values, learning_rate)

            # Compute loss and accuracy
            if epoch % 10 == 0:
                activations, _ = self.forward(X)
                loss = -np.mean(np.sum(y * np.log(activations[-1] + 1e-15), axis=1))
                predictions = np.argmax(activations[-1], axis=1)
                accuracy = np.mean(predictions == np.argmax(y, axis=1))
                print(f"Epoch {epoch}: Loss={loss:.4f}, Accuracy={accuracy:.2%}")

    def predict(self, X):
        """Predict class labels"""
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis=1)

# Example: MNIST-like data
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load data
digits = load_digits()
X = digits.data / 16.0  # Normalize
y = digits.target

# One-hot encode labels
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Train
nn = NeuralNetwork([64, 128, 64, 10])
nn.train(X_train, y_train, epochs=100, learning_rate=0.1, batch_size=32)

# Test
y_pred = nn.predict(X_test)
y_true = np.argmax(y_test, axis=1)
accuracy = np.mean(y_pred == y_true)
print(f"\nTest Accuracy: {accuracy:.2%}")
```

### 3. Backpropagation Derivation

**Goal**: Compute ∂L/∂W^(l) for all layers

**Chain rule**:
```
∂L/∂W^(l) = ∂L/∂z^(l) · ∂z^(l)/∂W^(l)
          = δ^(l) · (a^(l-1))^T
```

**Error terms** (δ):
```python
# Output layer (softmax + cross-entropy)
δ^(L) = a^(L) - y

# Hidden layer l
δ^(l) = (W^(l+1))^T δ^(l+1) ⊙ σ'(z^(l))
```

**Example**: 2-layer network
```python
def backprop_example():
    """
    Derive backprop for simple 2-layer network

    Architecture: Input (d) → Hidden (h) → Output (K)
    """

    # Forward pass
    # z1 = W1 @ x + b1    (h,)
    # a1 = σ(z1)          (h,)
    # z2 = W2 @ a1 + b2   (K,)
    # a2 = softmax(z2)    (K,)

    # Loss: Cross-entropy
    # L = -Σ y_k log(a2_k)

    # Backward pass
    # ∂L/∂z2 = a2 - y           (K,)
    # ∂L/∂W2 = (a2-y) @ a1^T    (K, h)
    # ∂L/∂b2 = a2 - y           (K,)

    # ∂L/∂a1 = W2^T @ (a2-y)    (h,)
    # ∂L/∂z1 = ∂L/∂a1 ⊙ σ'(z1)  (h,)
    # ∂L/∂W1 = ∂L/∂z1 @ x^T     (h, d)
    # ∂L/∂b1 = ∂L/∂z1            (h,)

    pass
```

### 4. PyTorch Basics

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Prepare data
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(np.argmax(y_train, axis=1))
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model
model = SimpleNet(input_size=64, hidden_size=128, num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test)
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == torch.LongTensor(y_true)).sum().item() / len(y_true)
    print(f'Test Accuracy: {accuracy:.2%}')
```

### 5. Activation Functions

```python
def compare_activations():
    """Compare different activation functions"""
    x = np.linspace(-5, 5, 100)

    # Sigmoid
    sigmoid = 1 / (1 + np.exp(-x))

    # Tanh
    tanh = np.tanh(x)

    # ReLU
    relu = np.maximum(0, x)

    # Leaky ReLU
    leaky_relu = np.where(x > 0, x, 0.01 * x)

    # Plot
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 4, 1)
    plt.plot(x, sigmoid)
    plt.title('Sigmoid')
    plt.grid(True)

    plt.subplot(1, 4, 2)
    plt.plot(x, tanh)
    plt.title('Tanh')
    plt.grid(True)

    plt.subplot(1, 4, 3)
    plt.plot(x, relu)
    plt.title('ReLU')
    plt.grid(True)

    plt.subplot(1, 4, 4)
    plt.plot(x, leaky_relu)
    plt.title('Leaky ReLU')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

compare_activations()
```

**Pros/Cons**:
```
Sigmoid:
  + Smooth, bounded [0,1]
  - Vanishing gradients
  - Not zero-centered

Tanh:
  + Zero-centered
  - Still vanishing gradients

ReLU:
  + Fast computation
  + No vanishing gradient for x>0
  - Dead neurons (gradient=0 for x<0)

Leaky ReLU:
  + Fixes dead ReLU problem
  + All above ReLU pros
```

### 6. Weight Initialization

```python
def weight_initialization_comparison():
    """Compare different initialization schemes"""

    n_in, n_out = 100, 50

    # Bad: All zeros (symmetry problem)
    W_zeros = np.zeros((n_in, n_out))

    # Bad: Large random values (exploding gradients)
    W_large = np.random.randn(n_in, n_out) * 10

    # Xavier/Glorot (for tanh/sigmoid)
    W_xavier = np.random.randn(n_in, n_out) * np.sqrt(1 / n_in)

    # He (for ReLU)
    W_he = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)

    # Compare variance of activations
    X = np.random.randn(1000, n_in)

    for name, W in [('Zeros', W_zeros),
                    ('Large', W_large),
                    ('Xavier', W_xavier),
                    ('He', W_he)]:
        z = X @ W
        print(f"{name:8s}: mean={np.mean(z):.3f}, std={np.std(z):.3f}")

weight_initialization_comparison()
```

### 7. Regularization Techniques

**Dropout**:
```python
class DropoutNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_p=0.5):
        super(DropoutNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Randomly zero out neurons
        x = self.fc2(x)
        return x

# During training: dropout active
model.train()
output = model(input)

# During test: dropout off (scale activations)
model.eval()
with torch.no_grad():
    output = model(input)
```

**Early Stopping**:
```python
def train_with_early_stopping(model, train_loader, val_loader, patience=5):
    """Stop training when validation loss stops improving"""
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(100):
        # Train
        model.train()
        for inputs, labels in train_loader:
            # ... training code ...
            pass

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    return model
```

## Practice Problems

### Problem 1: Derive Backprop for Single Neuron
f(x) = σ(w^T x + b), loss L = (y - f(x))²

Find ∂L/∂w and ∂L/∂b.

<details>
<summary>Solution</summary>

```python
# f = σ(z) where z = w^T x + b
# L = (y - f)²

# Chain rule:
# ∂L/∂w = ∂L/∂f · ∂f/∂z · ∂z/∂w

# ∂L/∂f = -2(y - f)
# ∂f/∂z = σ'(z) = σ(z)(1 - σ(z)) = f(1 - f)
# ∂z/∂w = x

# Therefore:
# ∂L/∂w = -2(y - f) · f(1 - f) · x
# ∂L/∂b = -2(y - f) · f(1 - f)

def gradient_single_neuron(x, y, w, b):
    z = np.dot(w, x) + b
    f = 1 / (1 + np.exp(-z))  # Sigmoid

    dL_df = -2 * (y - f)
    df_dz = f * (1 - f)

    dL_dw = dL_df * df_dz * x
    dL_db = dL_df * df_dz

    return dL_dw, dL_db
```
</details>

### Problem 2: Implement Batch Normalization
Normalize activations to have mean=0, std=1.

<details>
<summary>Solution</summary>

```python
class BatchNorm:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

        # Running statistics
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, X, training=True):
        if training:
            # Compute batch statistics
            mean = np.mean(X, axis=0)
            var = np.var(X, axis=0)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            # Use running statistics
            mean = self.running_mean
            var = self.running_var

        # Normalize
        X_norm = (X - mean) / np.sqrt(var + self.eps)

        # Scale and shift
        out = self.gamma * X_norm + self.beta

        return out
```
</details>

## Key Takeaways

1. **Forward pass**: Compute activations layer by layer
2. **Backpropagation**: Chain rule to compute gradients
3. **Activation functions**: ReLU most common for hidden layers
4. **Weight initialization**: Xavier/He prevents gradient issues
5. **Regularization**: Dropout, early stopping prevent overfitting
6. **PyTorch**: High-level framework for building networks

## Resources

- [Neural Networks and Deep Learning (Book)](http://neuralnetworksanddeeplearning.com/)
- [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
