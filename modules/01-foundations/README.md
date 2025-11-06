# Module 1: Foundations

## Overview
Mathematical foundations and core concepts for machine learning.

## Key Topics

### 1. Machine Learning Problem Setup

**Three types of learning**:
```
Supervised: Learn from labeled data (X, y)
- Regression: Predict continuous values
- Classification: Predict discrete labels

Unsupervised: Find patterns in unlabeled data
- Clustering: Group similar data
- Dimensionality reduction: Compress data

Reinforcement: Learn from rewards/penalties
- Agent interacts with environment
- Maximize cumulative reward
```

**ML Pipeline**:
```python
1. Collect data
2. Split into train/validation/test
3. Choose model (hypothesis class)
4. Define loss function
5. Optimize (training)
6. Evaluate on test set
7. Deploy
```

### 2. Linear Algebra Essentials

**Vectors and Matrices**:
```python
import numpy as np

# Vector operations
v = np.array([1, 2, 3])
w = np.array([4, 5, 6])

# Dot product
dot = np.dot(v, w)  # 1*4 + 2*5 + 3*6 = 32

# Norm
norm = np.linalg.norm(v)  # sqrt(1^2 + 2^2 + 3^2) = sqrt(14)

# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A @ B  # Matrix product

# Transpose
A_T = A.T

# Inverse (if exists)
A_inv = np.linalg.inv(A)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
```

**Key Concepts**:
- **Rank**: Max # of linearly independent rows/columns
- **Eigenvalues**: λ where Av = λv
- **Positive definite**: x^T A x > 0 for all x ≠ 0
- **Orthogonal**: Q^T Q = I

### 3. Calculus for ML

**Gradients**:
```python
# Gradient: vector of partial derivatives
# For f(x, y), gradient is [∂f/∂x, ∂f/∂y]

# Example: f(x, y) = x^2 + 2xy + y^2
def f(x, y):
    return x**2 + 2*x*y + y**2

def grad_f(x, y):
    df_dx = 2*x + 2*y
    df_dy = 2*x + 2*y
    return np.array([df_dx, df_dy])

# At point (1, 2)
gradient = grad_f(1, 2)  # [6, 6]
```

**Chain Rule**:
```
If y = f(g(x)), then dy/dx = (df/dg) * (dg/dx)

Critical for backpropagation!
```

**Jacobian and Hessian**:
```python
# Jacobian: matrix of first derivatives
# For f: R^n → R^m, J is m×n matrix
# J[i,j] = ∂f_i/∂x_j

# Hessian: matrix of second derivatives
# For f: R^n → R, H is n×n matrix
# H[i,j] = ∂²f/∂x_i∂x_j

def hessian_example():
    # f(x, y) = x^2 + xy + y^2
    # ∂²f/∂x² = 2, ∂²f/∂y² = 2, ∂²f/∂x∂y = 1
    H = np.array([[2, 1],
                  [1, 2]])
    return H
```

### 4. Probability Basics

**Distributions**:
```python
import scipy.stats as stats

# Gaussian (Normal) distribution
mu, sigma = 0, 1
normal = stats.norm(mu, sigma)

# Sample
samples = normal.rvs(size=1000)

# PDF at x
pdf_value = normal.pdf(0)  # 1/sqrt(2π)

# CDF
cdf_value = normal.cdf(0)  # 0.5

# Bernoulli (binary outcome)
p = 0.7
bernoulli = stats.bernoulli(p)

# Multinomial (categorical)
probs = [0.2, 0.3, 0.5]
samples = np.random.choice(3, size=100, p=probs)
```

**Key Concepts**:
- **Expectation**: E[X] = ∫ x p(x) dx
- **Variance**: Var[X] = E[(X - E[X])²]
- **Covariance**: Cov[X,Y] = E[(X-E[X])(Y-E[Y])]
- **Independence**: P(X,Y) = P(X)P(Y)
- **Bayes' Rule**: P(A|B) = P(B|A)P(A)/P(B)

```python
# Bayes' Rule example: Medical test
def bayes_theorem(prior, true_positive_rate, false_positive_rate):
    """
    Calculate P(disease | positive test)

    prior: P(disease)
    true_positive_rate: P(positive | disease)
    false_positive_rate: P(positive | no disease)
    """
    # P(positive) = P(positive|disease)P(disease) + P(positive|no disease)P(no disease)
    p_positive = (true_positive_rate * prior +
                  false_positive_rate * (1 - prior))

    # P(disease | positive) = P(positive|disease)P(disease) / P(positive)
    posterior = (true_positive_rate * prior) / p_positive

    return posterior

# Example: 1% disease prevalence, 99% TPR, 5% FPR
prob_disease = bayes_theorem(0.01, 0.99, 0.05)
print(f"P(disease | positive) = {prob_disease:.2%}")  # ~16.6%
```

### 5. Loss Functions

**Regression Loss**:
```python
# Mean Squared Error (L2 loss)
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Mean Absolute Error (L1 loss)
def mae_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Huber loss (robust to outliers)
def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small = np.abs(error) <= delta

    small_loss = 0.5 * error**2
    large_loss = delta * (np.abs(error) - 0.5 * delta)

    return np.mean(np.where(is_small, small_loss, large_loss))
```

**Classification Loss**:
```python
# 0-1 Loss (non-differentiable)
def zero_one_loss(y_true, y_pred):
    return np.mean(y_true != y_pred)

# Cross-entropy (log loss)
def cross_entropy(y_true, y_pred_prob):
    """
    y_true: true labels (0 or 1)
    y_pred_prob: predicted probabilities
    """
    epsilon = 1e-15  # Avoid log(0)
    y_pred_prob = np.clip(y_pred_prob, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred_prob) +
                    (1 - y_true) * np.log(1 - y_pred_prob))

# Hinge loss (SVM)
def hinge_loss(y_true, y_pred):
    """
    y_true: {-1, +1}
    y_pred: raw scores
    """
    return np.mean(np.maximum(0, 1 - y_true * y_pred))
```

### 6. Optimization Basics

**Gradient Descent**:
```python
def gradient_descent(f, grad_f, x0, learning_rate=0.01, max_iters=1000, tol=1e-6):
    """
    Minimize function f using gradient descent

    f: function to minimize
    grad_f: gradient of f
    x0: initial point
    learning_rate: step size
    max_iters: maximum iterations
    tol: convergence tolerance
    """
    x = x0.copy()
    history = [f(x)]

    for i in range(max_iters):
        # Compute gradient
        grad = grad_f(x)

        # Update: x ← x - α∇f(x)
        x_new = x - learning_rate * grad

        # Check convergence
        if np.linalg.norm(x_new - x) < tol:
            print(f"Converged in {i} iterations")
            break

        x = x_new
        history.append(f(x))

    return x, history

# Example: minimize f(x) = x^2 + 2x + 1
def f(x):
    return x**2 + 2*x + 1

def grad_f(x):
    return 2*x + 2

x_opt, history = gradient_descent(f, grad_f, x0=np.array([5.0]), learning_rate=0.1)
print(f"Minimum at x = {x_opt}")  # Should be -1
```

**Stochastic Gradient Descent (SGD)**:
```python
def sgd(X, y, model, loss_fn, grad_fn, epochs=10, batch_size=32, lr=0.01):
    """
    Stochastic Gradient Descent

    X: input data (n_samples, n_features)
    y: labels (n_samples,)
    model: parameters to optimize
    loss_fn: loss function
    grad_fn: gradient function
    """
    n_samples = len(X)

    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Mini-batch updates
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # Compute gradient on batch
            grad = grad_fn(model, X_batch, y_batch)

            # Update parameters
            model -= lr * grad

        # Compute loss after epoch
        loss = loss_fn(model, X, y)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    return model
```

### 7. Train/Val/Test Split

```python
from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split data into train/validation/test sets

    Common splits:
    - 60% train, 20% validation, 20% test
    - 70% train, 15% validation, 15% test
    """
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

# Example usage
X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, 1000)

X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
```

**Why three splits?**
- **Train**: Fit model parameters
- **Validation**: Tune hyperparameters (learning rate, regularization)
- **Test**: Final evaluation (never touch during development!)

### 8. Bias-Variance Tradeoff

```python
import matplotlib.pyplot as plt

def bias_variance_demo():
    """
    Demonstrate bias-variance tradeoff
    """
    # True function: y = sin(x) + noise
    np.random.seed(42)
    X_true = np.linspace(0, 10, 100)
    y_true = np.sin(X_true)

    # Training data (noisy)
    X_train = np.random.uniform(0, 10, 20)
    y_train = np.sin(X_train) + np.random.normal(0, 0.3, 20)

    # Fit polynomial models of different degrees
    degrees = [1, 4, 15]

    plt.figure(figsize=(15, 4))
    for i, degree in enumerate(degrees):
        # Fit polynomial
        coeffs = np.polyfit(X_train, y_train, degree)
        y_pred = np.polyval(coeffs, X_true)

        # Plot
        plt.subplot(1, 3, i+1)
        plt.scatter(X_train, y_train, label='Training data')
        plt.plot(X_true, y_true, 'g-', label='True function')
        plt.plot(X_true, y_pred, 'r-', label=f'Degree {degree}')
        plt.title(f'Polynomial degree {degree}')
        plt.legend()

    plt.tight_layout()
    plt.show()

# High bias (underfitting): Degree 1 - too simple
# Just right: Degree 4 - good fit
# High variance (overfitting): Degree 15 - too complex
```

**Key Insight**:
```
Total Error = Bias² + Variance + Irreducible Error

Bias: Error from wrong assumptions (underfitting)
Variance: Error from sensitivity to training data (overfitting)

Simple models: High bias, low variance
Complex models: Low bias, high variance
```

## Practice Problems

### Problem 1: Matrix Operations
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [1, 3]])

# Compute:
# a) A + B
# b) A @ B (matrix multiplication)
# c) Element-wise multiplication A * B
# d) Trace of A
# e) Determinant of A
```

<details>
<summary>Solution</summary>

```python
# a) Addition
A_plus_B = A + B  # [[3, 2], [4, 7]]

# b) Matrix multiplication
A_at_B = A @ B  # [[4, 6], [10, 12]]

# c) Element-wise
A_times_B = A * B  # [[2, 0], [3, 12]]

# d) Trace (sum of diagonal)
trace_A = np.trace(A)  # 1 + 4 = 5

# e) Determinant
det_A = np.linalg.det(A)  # 1*4 - 2*3 = -2
```
</details>

### Problem 2: Gradient Computation
Compute gradient of f(x, y) = x²y + y³ at point (2, 3).

<details>
<summary>Solution</summary>

```python
# f(x, y) = x²y + y³
# ∂f/∂x = 2xy
# ∂f/∂y = x² + 3y²

def grad_f(x, y):
    df_dx = 2*x*y
    df_dy = x**2 + 3*y**2
    return np.array([df_dx, df_dy])

gradient = grad_f(2, 3)
# df_dx = 2*2*3 = 12
# df_dy = 4 + 27 = 31
# Gradient: [12, 31]
```
</details>

### Problem 3: Loss Functions
Given y_true = [1, 0, 1, 1, 0] and y_pred = [0.9, 0.1, 0.8, 0.6, 0.3], compute cross-entropy loss.

<details>
<summary>Solution</summary>

```python
y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.6, 0.3])

# Cross-entropy: -mean(y*log(p) + (1-y)*log(1-p))
loss = -np.mean(
    y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
)

# Calculation:
# Sample 0: 1*log(0.9) = -0.105
# Sample 1: 0*log(0.1) + 1*log(0.9) = -0.105
# Sample 2: 1*log(0.8) = -0.223
# Sample 3: 1*log(0.6) = -0.511
# Sample 4: 0*log(0.3) + 1*log(0.7) = -0.357
# Mean = 0.260
```
</details>

## Key Takeaways

1. **Linear algebra**: Matrix operations, eigenvalues fundamental
2. **Calculus**: Gradients drive optimization
3. **Probability**: Models uncertainty, enables Bayesian thinking
4. **Loss functions**: Define what we optimize
5. **Optimization**: Gradient descent and variants
6. **Validation**: Proper data splitting prevents overfitting

## Resources

- Khan Academy: Linear Algebra, Calculus
- 3Blue1Brown: Visual linear algebra/calculus
- [Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)
