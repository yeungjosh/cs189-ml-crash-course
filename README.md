# CS 189: Introduction to Machine Learning - Crash Course

Comprehensive self-study guide for UC Berkeley's CS 189 (Introduction to Machine Learning).

## Course Overview

CS 189 covers theoretical foundations, algorithms, and applications of machine learning. Topics include supervised learning (regression, classification), unsupervised learning (clustering, dimensionality reduction), neural networks, deep learning, and generative models.

**Prerequisites**: Linear algebra (Math 54), multivariable calculus (Math 53), probability (CS 70)

## Repository Structure

```
modules/           # Crash course content by topic (one README per module)
├── 01-foundations/        # available
├── 02-linear-regression/  # available
├── 03-classification/     # available
├── 04-neural-networks/    # available
├── 05-unsupervised/       # planned
├── 06-deep-learning/      # planned
└── 07-advanced-topics/    # planned

resources/         # cheatsheet.md and links.md
```

Modules 5-7 and the homework guides (`assignments/`) listed below are outlines for future work and do not exist in the repo yet.

## Learning Path

### Module 1: Foundations (Week 1)
- Mathematical background (linear algebra, calculus, probability)
- ML problem formulation
- Loss functions and optimization
- Train/validation/test splits

### Module 2: Linear Regression (Week 2)
- Ordinary least squares
- Ridge regression (L2 regularization)
- Lasso (L1 regularization)
- Polynomial features
- Bias-variance tradeoff

### Module 3: Classification (Weeks 3-4)
- Perceptron algorithm
- Logistic regression
- Support Vector Machines (SVM)
- Decision trees and ensemble methods
- Gaussian discriminant analysis

### Module 4: Neural Networks (Week 5)
- Feedforward networks
- Backpropagation
- Activation functions
- Gradient descent variants
- PyTorch fundamentals

### Module 5: Unsupervised Learning (Week 6)
- K-means clustering
- Expectation-Maximization (EM)
- Principal Component Analysis (PCA)
- Dimensionality reduction

### Module 6: Deep Learning (Weeks 7-8)
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)
- Transformers and attention
- Batch normalization, dropout
- Transfer learning

### Module 7: Advanced Topics (Week 9)
- Generative models (VAE, GAN)
- Reinforcement learning basics
- Model interpretability
- Deployment considerations

## Assignments (planned outlines)

### HW1: Linear Models
Implement regression from scratch, experiment with regularization.

### HW2: Classification
Build classifiers (logistic regression, SVM), compare performance on real datasets.

### HW3: Neural Networks
Implement backpropagation, train networks on MNIST.

### HW4: Unsupervised Learning
K-means clustering, PCA for dimensionality reduction, visualize high-dimensional data.

### HW5: Deep Learning
CNNs for image classification (CIFAR-10), experiment with architectures.

## Study Tips

1. **Math foundations critical**: Review linear algebra and probability first
2. **Code from scratch**: Implement algorithms before using libraries
3. **Visualize**: Plot decision boundaries, loss curves, embeddings
4. **Real datasets**: Practice on UCI, Kaggle datasets
5. **Read papers**: Classic ML papers linked in resources

## Quick Start

```bash
# Clone repo
git clone https://github.com/yeungjosh/cs189-ml-crash-course.git
cd cs189-ml-crash-course

# Install dependencies
pip install numpy scipy matplotlib scikit-learn torch torchvision

# Start with Module 1: read modules/01-foundations/README.md
# (modules contain Markdown guides with code snippets; there are no notebooks yet)
```

## Resources

- **Textbooks**:
  - ISLR (James et al.) - [free online](https://www.statlearning.com/)
  - ESL (Hastie et al.) - [free online](https://hastie.su.domains/ElemStatLearn/)
  - Deep Learning (Bishop) - Springer
- **Course Website**: [eecs189.org](https://eecs189.org)
- **Lecture Notes**: [Jonathan Shewchuk's page](https://people.eecs.berkeley.edu/~jrs/189/)

## Tools & Libraries

- **NumPy**: Matrix operations, vectorization
- **Scikit-learn**: ML algorithms, evaluation metrics
- **PyTorch**: Deep learning framework
- **Matplotlib/Seaborn**: Visualization
- **Pandas**: Data manipulation

## Prerequisites Checklist

Before starting, you should understand:
- ✅ Matrix multiplication, eigenvalues/eigenvectors
- ✅ Gradients, partial derivatives, chain rule
- ✅ Probability distributions, expectation, variance
- ✅ Python programming, NumPy basics

## Contributing

Found errors or have improvements? Open an issue or PR.

## Note

This is an unofficial study guide. For official course materials, visit [eecs189.org](https://eecs189.org).
