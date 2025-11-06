# Module 2: Linear Regression

## Overview
Learn to predict continuous values using linear models, regularization, and feature engineering.

## Key Concepts

### 1. Simple Linear Regression

**Model**: y = w₀ + w₁x (line in 2D)

```python
import numpy as np
import matplotlib.pyplot as plt

class SimpleLinearRegression:
    def __init__(self):
        self.w0 = None  # Intercept
        self.w1 = None  # Slope

    def fit(self, X, y):
        """
        Fit using closed-form solution

        X: (n_samples,) input features
        y: (n_samples,) target values
        """
        # Add bias term
        n = len(X)
        X_mean = np.mean(X)
        y_mean = np.mean(y)

        # w1 = Cov(X,y) / Var(X)
        numerator = np.sum((X - X_mean) * (y - y_mean))
        denominator = np.sum((X - X_mean)**2)
        self.w1 = numerator / denominator

        # w0 = mean(y) - w1 * mean(X)
        self.w0 = y_mean - self.w1 * X_mean

    def predict(self, X):
        return self.w0 + self.w1 * X

    def score(self, X, y):
        """R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1 - (ss_res / ss_tot)

# Example
np.random.seed(42)
X = np.random.randn(100)
y = 2 + 3*X + np.random.randn(100)*0.5  # True: y = 2 + 3x + noise

model = SimpleLinearRegression()
model.fit(X, y)

print(f"Intercept: {model.w0:.2f}")  # Should be ~2
print(f"Slope: {model.w1:.2f}")      # Should be ~3
print(f"R²: {model.score(X, y):.2f}")

# Visualize
plt.scatter(X, y, alpha=0.5)
plt.plot(X, model.predict(X), 'r-', label='Fitted line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
```

### 2. Multiple Linear Regression

**Model**: y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ

**Matrix form**: y = Xw

```python
class LinearRegression:
    def __init__(self):
        self.w = None

    def fit(self, X, y):
        """
        Ordinary Least Squares (OLS)

        Minimize: ||y - Xw||²

        Closed-form solution: w = (X^T X)^(-1) X^T y
        """
        # Add intercept column
        X_bias = np.c_[np.ones(len(X)), X]

        # Compute (X^T X)^(-1) X^T y
        self.w = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y

    def predict(self, X):
        X_bias = np.c_[np.ones(len(X)), X]
        return X_bias @ self.w

    def mse(self, X, y):
        """Mean squared error"""
        y_pred = self.predict(X)
        return np.mean((y - y_pred)**2)

# Example: multiple features
n_samples, n_features = 200, 5
X = np.random.randn(n_samples, n_features)
w_true = np.array([1, 2, -1, 0.5, 3])
y = X @ w_true + np.random.randn(n_samples) * 0.1

model = LinearRegression()
model.fit(X, y)

print("True weights:", w_true)
print("Learned weights:", model.w[1:])  # Skip intercept
print("MSE:", model.mse(X, y))
```

### 3. Gradient Descent for Linear Regression

**Alternative to closed-form** (useful when X^T X not invertible):

```python
def gradient_descent_linear_regression(X, y, learning_rate=0.01, epochs=1000):
    """
    Minimize MSE using gradient descent

    Loss: L(w) = 1/n * ||y - Xw||²
    Gradient: ∇L = -2/n * X^T(y - Xw)
    """
    n, d = X.shape
    X_bias = np.c_[np.ones(n), X]
    w = np.zeros(d + 1)

    losses = []

    for epoch in range(epochs):
        # Predictions
        y_pred = X_bias @ w

        # Loss
        loss = np.mean((y - y_pred)**2)
        losses.append(loss)

        # Gradient
        gradient = -2/n * X_bias.T @ (y - y_pred)

        # Update
        w -= learning_rate * gradient

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return w, losses

# Example
X = np.random.randn(100, 3)
y = X @ [2, -1, 3] + np.random.randn(100) * 0.1

w_gd, losses = gradient_descent_linear_regression(X, y, learning_rate=0.1, epochs=500)

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss')
plt.show()
```

### 4. Ridge Regression (L2 Regularization)

**Problem**: OLS overfits with many features or collinear features

**Solution**: Add penalty for large weights

**Objective**: ||y - Xw||² + λ||w||²

```python
class RidgeRegression:
    def __init__(self, alpha=1.0):
        """
        alpha: regularization strength (λ)
        Higher alpha → stronger regularization → smaller weights
        """
        self.alpha = alpha
        self.w = None

    def fit(self, X, y):
        """
        Ridge solution: w = (X^T X + λI)^(-1) X^T y
        """
        X_bias = np.c_[np.ones(len(X)), X]
        n, d = X_bias.shape

        # Identity matrix (don't regularize intercept)
        I = np.eye(d)
        I[0, 0] = 0  # Don't penalize intercept

        # Solve
        self.w = np.linalg.inv(X_bias.T @ X_bias + self.alpha * I) @ X_bias.T @ y

    def predict(self, X):
        X_bias = np.c_[np.ones(len(X)), X]
        return X_bias @ self.w

# Compare OLS vs Ridge
X = np.random.randn(50, 20)  # 20 features, 50 samples (p > n)
w_true = np.concatenate([np.random.randn(5), np.zeros(15)])  # Sparse true weights
y = X @ w_true + np.random.randn(50) * 0.5

# OLS (may overfit)
ols = LinearRegression()
ols.fit(X, y)

# Ridge (regularized)
ridge = RidgeRegression(alpha=10.0)
ridge.fit(X, y)

# Test set
X_test = np.random.randn(100, 20)
y_test = X_test @ w_true + np.random.randn(100) * 0.5

print(f"OLS Test MSE: {ols.mse(X_test, y_test):.4f}")
print(f"Ridge Test MSE: {ridge.predict(X_test); np.mean((y_test - ridge.predict(X_test))**2):.4f}")

# Plot weights
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.bar(range(20), w_true)
plt.title('True Weights')

plt.subplot(1, 3, 2)
plt.bar(range(20), ols.w[1:])
plt.title('OLS Weights')

plt.subplot(1, 3, 3)
plt.bar(range(20), ridge.w[1:])
plt.title('Ridge Weights (α=10)')

plt.tight_layout()
plt.show()
```

### 5. Lasso Regression (L1 Regularization)

**Objective**: ||y - Xw||² + λ||w||₁

**Benefit**: Produces sparse weights (many exactly 0)

```python
from sklearn.linear_model import Lasso  # Use sklearn (coordinate descent)

class LassoDemo:
    @staticmethod
    def compare_regularization():
        """Compare Ridge vs Lasso"""
        # Generate sparse data
        n, d = 100, 50
        X = np.random.randn(n, d)
        w_true = np.zeros(d)
        w_true[:5] = [5, -3, 2, -4, 1]  # Only 5 non-zero
        y = X @ w_true + np.random.randn(n)

        # Fit models
        ridge = RidgeRegression(alpha=1.0)
        lasso = Lasso(alpha=1.0)

        ridge.fit(X, y)
        lasso.fit(X, y)

        # Compare sparsity
        ridge_nonzero = np.sum(np.abs(ridge.w[1:]) > 0.01)
        lasso_nonzero = np.sum(np.abs(lasso.coef_) > 0.01)

        print(f"Ridge non-zero weights: {ridge_nonzero}/50")
        print(f"Lasso non-zero weights: {lasso_nonzero}/50")  # Much fewer!

        # Visualize
        plt.figure(figsize=(15, 4))

        plt.subplot(1, 3, 1)
        plt.bar(range(d), w_true)
        plt.title('True Weights (5 non-zero)')

        plt.subplot(1, 3, 2)
        plt.bar(range(d), ridge.w[1:])
        plt.title(f'Ridge ({ridge_nonzero} non-zero)')

        plt.subplot(1, 3, 3)
        plt.bar(range(d), lasso.coef_)
        plt.title(f'Lasso ({lasso_nonzero} non-zero)')

        plt.tight_layout()
        plt.show()

LassoDemo.compare_regularization()
```

**Key Difference**:
```
Ridge (L2): Shrinks all weights, keeps all features
Lasso (L1): Zeros out weights, performs feature selection
```

### 6. Polynomial Features

**Extend linear models** to capture non-linear relationships:

```python
def polynomial_features(X, degree):
    """
    Create polynomial features

    degree=2: [1, x, x²]
    degree=3: [1, x, x², x³]
    """
    n = len(X)
    X_poly = np.ones((n, degree + 1))

    for d in range(1, degree + 1):
        X_poly[:, d] = X ** d

    return X_poly

# Example: fit y = x² + noise with linear model
X = np.linspace(-3, 3, 100)
y = X**2 + np.random.randn(100) * 0.5

# Linear fit (bad)
X_linear = X.reshape(-1, 1)
model_linear = LinearRegression()
model_linear.fit(X_linear, y)
y_pred_linear = model_linear.predict(X_linear)

# Quadratic fit (good)
X_quad = polynomial_features(X, degree=2)
model_quad = LinearRegression()
model_quad.fit(X_quad[:, 1:], y)  # Exclude intercept column
y_pred_quad = model_quad.predict(X_quad[:, 1:])

# Cubic fit (may overfit)
X_cubic = polynomial_features(X, degree=10)
model_cubic = LinearRegression()
model_cubic.fit(X_cubic[:, 1:], y)
y_pred_cubic = model_cubic.predict(X_cubic[:, 1:])

# Plot
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.scatter(X, y, alpha=0.5)
plt.plot(X, y_pred_linear, 'r-', linewidth=2)
plt.title('Linear (degree=1)')

plt.subplot(1, 3, 2)
plt.scatter(X, y, alpha=0.5)
plt.plot(X, y_pred_quad, 'r-', linewidth=2)
plt.title('Quadratic (degree=2)')

plt.subplot(1, 3, 3)
plt.scatter(X, y, alpha=0.5)
plt.plot(X, y_pred_cubic, 'r-', linewidth=2)
plt.title('High-degree (degree=10)')

plt.tight_layout()
plt.show()
```

### 7. Model Selection and Cross-Validation

```python
from sklearn.model_selection import KFold

def cross_validate(X, y, model_class, k_folds=5, **model_params):
    """
    K-fold cross-validation

    Split data into k parts, train on k-1, validate on 1
    Repeat k times, average results
    """
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train model
        model = model_class(**model_params)
        model.fit(X_train, y_train)

        # Evaluate
        mse = np.mean((y_val - model.predict(X_val))**2)
        scores.append(mse)

    return np.mean(scores), np.std(scores)

# Example: choose best regularization strength
X = np.random.randn(100, 10)
y = X @ np.random.randn(10) + np.random.randn(100) * 0.5

alphas = [0.01, 0.1, 1, 10, 100]
results = []

for alpha in alphas:
    mean_mse, std_mse = cross_validate(X, y, RidgeRegression, k_folds=5, alpha=alpha)
    results.append((alpha, mean_mse, std_mse))
    print(f"α={alpha:6.2f}: MSE={mean_mse:.4f} ± {std_mse:.4f}")

# Choose best alpha
best_alpha = min(results, key=lambda x: x[1])[0]
print(f"\nBest α: {best_alpha}")
```

## Practice Problems

### Problem 1: Implement Standardization
Standardize features to have mean=0, std=1. Why important?

<details>
<summary>Solution</summary>

```python
def standardize(X):
    """
    Standardize features: (X - mean) / std

    Why?
    - Gradient descent converges faster
    - Regularization treats features equally
    - Numerical stability
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std, mean, std

def unstandardize_weights(w, mean, std):
    """Convert weights back to original scale"""
    w_original = w[1:] / std
    w0_original = w[0] - np.sum(w[1:] * mean / std)
    return np.concatenate([[w0_original], w_original])

# Usage
X = np.random.randn(100, 5) * 10 + 5  # Different scales
X_std, mean, std = standardize(X)

print("Original mean:", X.mean(axis=0))
print("Standardized mean:", X_std.mean(axis=0))  # ~0
print("Standardized std:", X_std.std(axis=0))    # ~1
```
</details>

### Problem 2: Regularization Path
Plot how Ridge coefficients change with λ.

<details>
<summary>Solution</summary>

```python
def ridge_path(X, y, alphas):
    """Plot coefficient paths"""
    n, d = X.shape
    X_bias = np.c_[np.ones(n), X]

    coeffs = []
    for alpha in alphas:
        I = np.eye(d + 1)
        I[0, 0] = 0
        w = np.linalg.inv(X_bias.T @ X_bias + alpha * I) @ X_bias.T @ y
        coeffs.append(w[1:])  # Exclude intercept

    coeffs = np.array(coeffs)

    # Plot
    plt.figure(figsize=(10, 6))
    for i in range(d):
        plt.plot(alphas, coeffs[:, i], label=f'w_{i+1}')

    plt.xscale('log')
    plt.xlabel('λ (log scale)')
    plt.ylabel('Coefficient value')
    plt.title('Ridge Regularization Path')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example
X = np.random.randn(50, 5)
y = X @ [3, -2, 1, -1, 2] + np.random.randn(50) * 0.1
alphas = np.logspace(-3, 3, 50)

ridge_path(X, y, alphas)
```
</details>

### Problem 3: Learning Curves
Plot train/val error vs training set size.

<details>
<summary>Solution</summary>

```python
def plot_learning_curves(X, y, model_class, **model_params):
    """
    Learning curve: error vs training set size

    High bias: Both errors high, plateau
    High variance: Large gap between train/val error
    """
    n = len(X)
    sizes = range(10, n, 10)

    train_errors = []
    val_errors = []

    for size in sizes:
        # Split
        X_train, X_val = X[:size], X[size:]
        y_train, y_val = y[:size], y[size:]

        # Train
        model = model_class(**model_params)
        model.fit(X_train, y_train)

        # Errors
        train_mse = model.mse(X_train, y_train)
        val_mse = model.mse(X_val, y_val)

        train_errors.append(train_mse)
        val_errors.append(val_mse)

    # Plot
    plt.plot(sizes, train_errors, label='Train Error')
    plt.plot(sizes, val_errors, label='Val Error')
    plt.xlabel('Training Set Size')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('Learning Curves')
    plt.show()
```
</details>

## Key Takeaways

1. **OLS**: Closed-form solution w = (X^T X)^(-1) X^T y
2. **Ridge**: L2 regularization shrinks weights
3. **Lasso**: L1 regularization selects features
4. **Polynomial features**: Linear models for non-linear data
5. **Cross-validation**: Proper model selection
6. **Bias-variance**: Balance complexity

## Resources

- ISLR Chapter 3: Linear Regression
- ESL Chapter 3: Linear Methods for Regression
- Scikit-learn: [Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
