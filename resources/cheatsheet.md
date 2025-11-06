# CS 189 ML Cheat Sheet

## Linear Regression

```python
# Ordinary Least Squares
w = (X^T X)^(-1) X^T y

# Ridge (L2 regularization)
w = (X^T X + λI)^(-1) X^T y

# Gradient descent update
w ← w - α∇L(w)
∇L = -2/n X^T(y - Xw)

# MSE loss
L = 1/n ||y - Xw||²
```

## Classification

### Logistic Regression
```python
# Model
P(y=1|x) = σ(w^T x)  where σ(z) = 1/(1 + e^(-z))

# Cross-entropy loss
L = -1/n Σ[y log(σ(w^T x)) + (1-y) log(1-σ(w^T x))]

# Gradient
∇L = -1/n X^T(y - σ(Xw))
```

### SVM
```python
# Objective
min 1/2||w||² + C Σ max(0, 1 - y_i(w^T x_i + b))

# Kernels
Linear: K(x, x') = x^T x'
Polynomial: K(x, x') = (x^T x' + c)^d
RBF: K(x, x') = exp(-γ||x - x'||²)
```

### Softmax (Multi-class)
```python
# Probabilities
P(y=k|x) = exp(w_k^T x) / Σ_j exp(w_j^T x)

# Cross-entropy loss
L = -1/n ΣΣ 1{y_i=k} log P(y=k|x_i)
```

## Neural Networks

### Forward Pass
```python
# Layer l
z^(l) = W^(l) a^(l-1) + b^(l)
a^(l) = σ(z^(l))

# Common activations
ReLU: max(0, z)
Sigmoid: 1/(1 + e^(-z))
Tanh: (e^z - e^(-z))/(e^z + e^(-z))
```

### Backpropagation
```python
# Output layer
δ^(L) = ∇_a L ⊙ σ'(z^(L))

# Hidden layers
δ^(l) = (W^(l+1))^T δ^(l+1) ⊙ σ'(z^(l))

# Gradients
∇_{W^(l)} L = δ^(l) (a^(l-1))^T
∇_{b^(l)} L = δ^(l)

# Update
W ← W - α∇_W L
```

## Unsupervised Learning

### K-means
```python
# Algorithm
1. Initialize K centroids
2. Assign points to nearest centroid
3. Update centroids = mean of assigned points
4. Repeat 2-3 until convergence

# Objective
min Σ_k Σ_{x∈C_k} ||x - μ_k||²
```

### PCA
```python
# Steps
1. Center data: X_centered = X - mean(X)
2. Compute covariance: Σ = X^T X / n
3. Eigendecomposition: Σ = VΛV^T
4. Project: Z = X @ V[:, :k]  # Keep top k eigenvectors

# Explained variance
Σ_{i=1}^k λ_i / Σ_{i=1}^d λ_i
```

## Key Formulas

### Matrix Calculus
```python
# Scalar by vector
∂(a^T x)/∂x = a
∂(x^T A x)/∂x = (A + A^T)x

# Vector by vector (Jacobian)
∂f/∂x = [∂f_i/∂x_j]  # Matrix

# Common derivatives
∂||Xw - y||²/∂w = 2X^T(Xw - y)
∂(w^T w)/∂w = 2w
```

### Probability
```python
# Bayes' Rule
P(A|B) = P(B|A)P(A) / P(B)

# Gaussian
p(x|μ,Σ) = (2π)^(-d/2) |Σ|^(-1/2) exp(-1/2(x-μ)^T Σ^(-1)(x-μ))

# MLE for Gaussian
μ_MLE = 1/n Σ x_i
Σ_MLE = 1/n Σ(x_i - μ)(x_i - μ)^T
```

## Bias-Variance Tradeoff

```
Total Error = Bias² + Variance + Irreducible Error

Bias = E[f_hat(x)] - f(x)
Variance = E[(f_hat(x) - E[f_hat(x)])²]

High capacity → Low bias, high variance (overfit)
Low capacity → High bias, low variance (underfit)
```

## Regularization

```python
# L2 (Ridge)
Loss = MSE + λ||w||²

# L1 (Lasso)
Loss = MSE + λ||w||₁

# Elastic Net
Loss = MSE + λ₁||w||₁ + λ₂||w||²

# Dropout (Neural nets)
During training: Randomly set activations to 0 with prob p
During test: Scale activations by (1-p)
```

## Evaluation Metrics

### Regression
```python
MSE = 1/n Σ(y_i - ŷ_i)²
MAE = 1/n Σ|y_i - ŷ_i|
R² = 1 - SS_res/SS_tot
```

### Classification
```python
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 × (Precision × Recall) / (Precision + Recall)

# Multi-class
Accuracy = 1/n Σ 1{y_i = ŷ_i}
```

## Optimization

### Gradient Descent Variants
```python
# Batch GD
w ← w - α∇L(w)

# SGD (single sample)
w ← w - α∇L_i(w)

# Mini-batch SGD
w ← w - α∇L_batch(w)

# Momentum
v ← βv - α∇L
w ← w + v

# Adam
m ← β₁m + (1-β₁)∇L
v ← β₂v + (1-β₂)(∇L)²
w ← w - α m/(√v + ε)
```

### Learning Rate Schedules
```python
# Step decay
α_t = α_0 × γ^(t/k)

# Exponential decay
α_t = α_0 × e^(-kt)

# 1/t decay
α_t = α_0 / (1 + kt)
```

## PyTorch Quick Reference

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training loop
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    for batch_X, batch_y in dataloader:
        # Forward
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Scikit-learn Quick Reference

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix

# Standard workflow
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Cross-validation
scores = cross_val_score(model, X, y, cv=5)
```

## Common Numpy Operations

```python
import numpy as np

# Matrix ops
A @ B           # Matrix multiplication
A * B           # Element-wise
A.T             # Transpose
np.linalg.inv(A)  # Inverse
np.linalg.eig(A)  # Eigenvalues/vectors
np.trace(A)     # Trace
np.linalg.det(A)  # Determinant

# Statistics
np.mean(A, axis=0)  # Column means
np.std(A, axis=1)   # Row stds
np.cov(X.T)         # Covariance matrix

# Useful functions
np.exp(x)
np.log(x)
np.sqrt(x)
np.maximum(0, x)  # ReLU
np.argmax(x, axis=1)
np.clip(x, min, max)
```

## Decision Tree Metrics

```python
# Entropy
H(Y) = -Σ p(y) log₂ p(y)

# Information Gain
IG = H(parent) - Σ (|child|/|parent|) H(child)

# Gini Impurity
Gini = 1 - Σ p_i²
```

## Dimensionality Reduction

```python
# PCA: Maximize variance
Keep top k eigenvectors of covariance matrix

# LDA: Maximize class separation
Between-class scatter / Within-class scatter

# t-SNE: Preserve local structure
Good for visualization, not feature extraction
```

## Ensemble Methods

```python
# Bagging: Bootstrap + Aggregate
Train on random subsets, average predictions
Example: Random Forest

# Boosting: Sequential training
Each model corrects previous mistakes
Examples: AdaBoost, Gradient Boosting, XGBoost

# Stacking: Meta-learning
Train meta-model on base model predictions
```

## Important Inequalities

```python
# Jensen's Inequality (convex f)
f(E[X]) ≤ E[f(X)]

# Cauchy-Schwarz
|x^T y| ≤ ||x|| ||y||

# Triangle Inequality
||x + y|| ≤ ||x|| + ||y||
```

## Activation Functions

```python
# ReLU
f(x) = max(0, x)
f'(x) = 1{x > 0}

# Sigmoid
f(x) = 1/(1 + e^(-x))
f'(x) = f(x)(1 - f(x))

# Tanh
f(x) = tanh(x)
f'(x) = 1 - f(x)²

# Softmax
f(x)_i = e^(x_i) / Σ e^(x_j)
```

## Normalization Techniques

```python
# Standardization (Z-score)
x' = (x - μ) / σ

# Min-Max Scaling
x' = (x - min) / (max - min)

# Batch Normalization
x' = γ(x - μ_batch)/σ_batch + β
```

## Hyperparameter Tuning

```python
# Common hyperparameters
Learning rate α
Regularization λ
Batch size
Number of epochs
Network architecture (layers, units)

# Tuning methods
Grid search
Random search
Bayesian optimization
```

## Remember

- **Always standardize features** before training
- **Use cross-validation** for model selection
- **Check for data leakage** (test set in training)
- **Start simple** before adding complexity
- **Monitor both train and val loss** for overfitting
- **Regularization** when overfitting
- **More data** usually helps more than fancier models
