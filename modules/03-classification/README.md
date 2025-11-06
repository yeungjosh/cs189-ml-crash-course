# Module 3: Classification

## Overview
Predict discrete labels (binary or multi-class) using linear and non-linear classifiers.

## Key Concepts

### 1. Logistic Regression

**Binary classification**: y ∈ {0, 1}

**Model**: P(y=1|x) = σ(w^T x) where σ(z) = 1/(1+e^(-z))

```python
class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.w = None

    def sigmoid(self, z):
        """Sigmoid function: σ(z) = 1 / (1 + e^(-z))"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip for numerical stability

    def fit(self, X, y):
        """
        Train using gradient descent

        Loss: Cross-entropy
        L(w) = -1/n Σ[y log(σ(w^T x)) + (1-y) log(1-σ(w^T x))]

        Gradient: ∇L = -1/n X^T (y - σ(Xw))
        """
        n, d = X.shape
        X_bias = np.c_[np.ones(n), X]
        self.w = np.zeros(d + 1)

        for epoch in range(self.epochs):
            # Forward pass
            z = X_bias @ self.w
            probs = self.sigmoid(z)

            # Gradient
            gradient = -1/n * X_bias.T @ (y - probs)

            # Update
            self.w -= self.lr * gradient

            # Loss (optional monitoring)
            if epoch % 100 == 0:
                loss = -np.mean(y * np.log(probs + 1e-15) +
                                (1 - y) * np.log(1 - probs + 1e-15))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict_proba(self, X):
        """Return probabilities"""
        X_bias = np.c_[np.ones(len(X)), X]
        return self.sigmoid(X_bias @ self.w)

    def predict(self, X, threshold=0.5):
        """Return binary predictions"""
        return (self.predict_proba(X) >= threshold).astype(int)

    def accuracy(self, X, y):
        """Classification accuracy"""
        return np.mean(self.predict(X) == y)

# Example: binary classification
np.random.seed(42)
n_samples = 200

# Class 0: centered at (-1, -1)
X0 = np.random.randn(n_samples//2, 2) + [-1, -1]
y0 = np.zeros(n_samples//2)

# Class 1: centered at (1, 1)
X1 = np.random.randn(n_samples//2, 2) + [1, 1]
y1 = np.ones(n_samples//2)

X = np.vstack([X0, X1])
y = np.concatenate([y0, y1])

# Train
model = LogisticRegression(learning_rate=0.1, epochs=500)
model.fit(X, y)

print(f"Accuracy: {model.accuracy(X, y):.2%}")

# Decision boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='k')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Logistic Regression Decision Boundary')
    plt.show()

plot_decision_boundary(model, X, y)
```

### 2. Support Vector Machines (SVM)

**Idea**: Find hyperplane that maximizes margin between classes

**Linear SVM**:
- Minimize: (1/2)||w||² subject to y_i(w^T x_i + b) ≥ 1
- Equivalent: Minimize (1/2)||w||² + C Σ max(0, 1 - y_i(w^T x_i + b))

```python
from sklearn.svm import SVC

class SVMDemo:
    @staticmethod
    def linear_svm():
        """Linear SVM example"""
        # Generate linearly separable data
        np.random.seed(42)
        X0 = np.random.randn(50, 2) + [-2, -2]
        X1 = np.random.randn(50, 2) + [2, 2]
        X = np.vstack([X0, X1])
        y = np.concatenate([np.zeros(50), np.ones(50)])

        # Train SVM
        svm = SVC(kernel='linear', C=1.0)
        svm.fit(X, y)

        print(f"Accuracy: {svm.score(X, y):.2%}")
        print(f"Support vectors: {len(svm.support_vectors_)}")

        # Plot decision boundary and margin
        plt.figure(figsize=(10, 5))

        # Decision boundary
        w = svm.coef_[0]
        b = svm.intercept_[0]
        x_line = np.linspace(-5, 5, 100)
        y_line = -(w[0] * x_line + b) / w[1]

        # Margins
        margin = 1 / np.sqrt(np.sum(w**2))
        y_up = y_line + margin
        y_down = y_line - margin

        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='k')
        plt.plot(x_line, y_line, 'k-', label='Decision boundary')
        plt.plot(x_line, y_up, 'k--', alpha=0.5, label='Margin')
        plt.plot(x_line, y_down, 'k--', alpha=0.5)

        # Support vectors
        plt.scatter(svm.support_vectors_[:, 0],
                   svm.support_vectors_[:, 1],
                   s=200, facecolors='none', edgecolors='k', linewidths=2,
                   label='Support vectors')

        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.title('Linear SVM')
        plt.show()

    @staticmethod
    def kernel_svm():
        """Non-linear SVM with RBF kernel"""
        # Generate non-linearly separable data (circles)
        from sklearn.datasets import make_circles
        X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)

        # Linear SVM (poor fit)
        svm_linear = SVC(kernel='linear')
        svm_linear.fit(X, y)

        # RBF kernel SVM (good fit)
        svm_rbf = SVC(kernel='rbf', gamma=1)
        svm_rbf.fit(X, y)

        print(f"Linear SVM accuracy: {svm_linear.score(X, y):.2%}")
        print(f"RBF SVM accuracy: {svm_rbf.score(X, y):.2%}")

        # Plot both
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        for ax, model, title in zip(axes,
                                     [svm_linear, svm_rbf],
                                     ['Linear Kernel', 'RBF Kernel']):
            # Decision boundary
            xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 100),
                               np.linspace(-1.5, 1.5, 100))
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='k')
            ax.set_title(title)

        plt.tight_layout()
        plt.show()

SVMDemo.linear_svm()
SVMDemo.kernel_svm()
```

**Common Kernels**:
```python
# Linear: K(x, x') = x^T x'
# Polynomial: K(x, x') = (x^T x' + c)^d
# RBF (Gaussian): K(x, x') = exp(-γ||x - x'||²)
```

### 3. Decision Trees

**Idea**: Recursively split data based on features

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

class DecisionTreeDemo:
    @staticmethod
    def train_and_visualize():
        """Train decision tree and visualize"""
        # Generate data
        np.random.seed(42)
        X = np.random.randn(200, 2)
        y = ((X[:, 0] > 0) & (X[:, 1] > 0) |
             (X[:, 0] < 0) & (X[:, 1] < 0)).astype(int)

        # Train tree
        tree = DecisionTreeClassifier(max_depth=3, random_state=42)
        tree.fit(X, y)

        print(f"Accuracy: {tree.score(X, y):.2%}")

        # Visualize tree
        plt.figure(figsize=(20, 10))
        plot_tree(tree, filled=True, feature_names=['x1', 'x2'],
                  class_names=['0', '1'], fontsize=10)
        plt.title('Decision Tree')
        plt.show()

        # Decision boundary
        xx, yy = np.meshgrid(np.linspace(-3, 3, 100),
                            np.linspace(-3, 3, 100))
        Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='k')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Decision Tree Boundary')
        plt.show()

DecisionTreeDemo.train_and_visualize()
```

**Information Gain** (splitting criterion):
```python
def entropy(y):
    """Shannon entropy: H = -Σ p_i log(p_i)"""
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-15))

def information_gain(y_parent, y_left, y_right):
    """IG = H(parent) - [|left|/|total| * H(left) + |right|/|total| * H(right)]"""
    n = len(y_parent)
    n_left, n_right = len(y_left), len(y_right)

    ig = entropy(y_parent) - (n_left/n * entropy(y_left) +
                               n_right/n * entropy(y_right))
    return ig

# Example
y_parent = [0, 0, 1, 1, 1, 1]
y_left = [0, 0, 1]
y_right = [1, 1, 1]

print(f"Entropy(parent): {entropy(y_parent):.3f}")
print(f"Information Gain: {information_gain(y_parent, y_left, y_right):.3f}")
```

### 4. Ensemble Methods

**Random Forest**: Multiple decision trees, average predictions

```python
from sklearn.ensemble import RandomForestClassifier

def random_forest_demo():
    """Random forest vs single tree"""
    # Generate data
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=300, noise=0.3, random_state=42)

    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Single tree
    tree = DecisionTreeClassifier(max_depth=5)
    tree.fit(X_train, y_train)

    # Random forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)

    print(f"Decision Tree Test Acc: {tree.score(X_test, y_test):.2%}")
    print(f"Random Forest Test Acc: {rf.score(X_test, y_test):.2%}")

    # Plot decision boundaries
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for ax, model, title in zip(axes,
                                 [tree, rf],
                                 ['Single Tree', 'Random Forest (100 trees)']):
        xx, yy = np.meshgrid(np.linspace(-2, 3, 100),
                            np.linspace(-2, 2, 100))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                   cmap='RdYlBu', edgecolors='k', alpha=0.7)
        ax.set_title(title)

    plt.tight_layout()
    plt.show()

random_forest_demo()
```

### 5. Multi-class Classification

**One-vs-Rest** (OvR): Train K binary classifiers

```python
class OneVsRestClassifier:
    def __init__(self, base_classifier):
        """
        base_classifier: binary classifier class (e.g., LogisticRegression)
        """
        self.base_classifier = base_classifier
        self.classifiers = []
        self.classes = None

    def fit(self, X, y):
        """Train K binary classifiers"""
        self.classes = np.unique(y)
        self.classifiers = []

        for c in self.classes:
            # Binary labels: 1 if class c, else 0
            y_binary = (y == c).astype(int)

            # Train classifier
            clf = self.base_classifier()
            clf.fit(X, y_binary)
            self.classifiers.append(clf)

    def predict_proba(self, X):
        """Return probabilities for each class"""
        probs = np.zeros((len(X), len(self.classes)))

        for i, clf in enumerate(self.classifiers):
            probs[:, i] = clf.predict_proba(X)

        # Normalize
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        """Return predicted class"""
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]

# Example: 3-class problem
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)

# Train
clf = OneVsRestClassifier(LogisticRegression)
clf.fit(X, y)

# Evaluate
accuracy = np.mean(clf.predict(X) == y)
print(f"Accuracy: {accuracy:.2%}")

# Visualize
xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100),
                     np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 100))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title('One-vs-Rest Multi-class Classification')
plt.show()
```

### 6. Evaluation Metrics

```python
def classification_metrics(y_true, y_pred):
    """
    Compute common classification metrics
    """
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)

    # True Positives, False Positives, False Negatives, True Negatives
    TP = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TN = cm[0, 0]

    # Metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(cm)

    return accuracy, precision, recall, f1

# Example
y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 0])

classification_metrics(y_true, y_pred)
```

**ROC Curve** (Receiver Operating Characteristic):
```python
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_scores):
    """
    Plot ROC curve

    y_true: true binary labels
    y_scores: predicted probabilities or scores
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example
model = LogisticRegression(epochs=500)
model.fit(X_train, y_train)
y_scores = model.predict_proba(X_test)

plot_roc_curve(y_test, y_scores)
```

## Practice Problems

### Problem 1: Implement Perceptron
Simple online learning algorithm.

<details>
<summary>Solution</summary>

```python
class Perceptron:
    def __init__(self, epochs=100):
        self.epochs = epochs
        self.w = None

    def fit(self, X, y):
        """
        Perceptron algorithm

        For each sample:
          If misclassified: w ← w + y_i * x_i
        """
        n, d = X.shape
        X_bias = np.c_[np.ones(n), X]
        self.w = np.zeros(d + 1)

        y_signed = 2*y - 1  # Convert {0,1} to {-1,+1}

        for epoch in range(self.epochs):
            mistakes = 0

            for i in range(n):
                pred = np.sign(X_bias[i] @ self.w)

                if pred * y_signed[i] <= 0:  # Misclassified
                    self.w += y_signed[i] * X_bias[i]
                    mistakes += 1

            if mistakes == 0:
                print(f"Converged in {epoch+1} epochs")
                break

    def predict(self, X):
        X_bias = np.c_[np.ones(len(X)), X]
        return (np.sign(X_bias @ self.w) + 1) / 2  # Back to {0,1}
```
</details>

### Problem 2: Softmax Regression
Multi-class logistic regression.

<details>
<summary>Solution</summary>

```python
def softmax(z):
    """
    Softmax: exp(z_i) / Σ exp(z_j)

    Numerically stable version
    """
    z_shifted = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.W = None  # Weight matrix (d+1, K)

    def fit(self, X, y):
        n, d = X.shape
        K = len(np.unique(y))  # Number of classes

        X_bias = np.c_[np.ones(n), X]
        self.W = np.zeros((d + 1, K))

        # One-hot encode labels
        Y = np.eye(K)[y]

        for epoch in range(self.epochs):
            # Forward
            logits = X_bias @ self.W
            probs = softmax(logits)

            # Gradient
            gradient = X_bias.T @ (probs - Y) / n

            # Update
            self.W -= self.lr * gradient

    def predict_proba(self, X):
        X_bias = np.c_[np.ones(len(X)), X]
        return softmax(X_bias @ self.W)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
```
</details>

## Key Takeaways

1. **Logistic Regression**: Linear decision boundary, probabilistic
2. **SVM**: Maximum margin, kernel trick for non-linearity
3. **Decision Trees**: Non-parametric, interpretable
4. **Random Forests**: Ensemble reduces variance
5. **Metrics**: Accuracy, precision, recall, F1, ROC-AUC

## Resources

- ISLR Chapter 4: Classification
- ESL Chapter 4: Linear Methods for Classification
- Scikit-learn: [Classification](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
