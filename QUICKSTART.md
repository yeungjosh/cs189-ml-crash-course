# CS 189 Quick Start Guide

## For Self-Learners

This crash course assumes junior SWE knowledge: data structures, algorithms, Python programming.

## Prerequisites

**Must have**:
- Linear algebra (matrices, eigenvalues, SVD)
- Calculus (gradients, chain rule)
- Probability (distributions, expectation, Bayes' rule)
- Python + NumPy

**Helpful**:
- Statistics
- Optimization theory
- PyTorch/TensorFlow basics

## Learning Path Options

### Fast Track (3-4 weeks)

**Goal**: Core ML concepts + practical implementation

**Week 1**: Foundations + Linear Regression
- Module 1: Math review
- Module 2: OLS, Ridge, Lasso
- Code: Implement from scratch

**Week 2**: Classification
- Module 3: Logistic regression, SVM, trees
- Practice: Scikit-learn on real datasets

**Week 3**: Neural Networks
- Module 4: Feedforward, backprop
- PyTorch basics

**Week 4**: Projects
- MNIST digit classification
- Kaggle competition

**Skip**: Deep learning advanced topics (focus on fundamentals)

### Deep Dive (8-10 weeks)

**Goal**: Comprehensive understanding + research readiness

**Weeks 1-2**: Foundations
- Module 1 thoroughly
- Linear algebra practice
- Implement gradient descent variants

**Weeks 3-4**: Regression + Classification
- Modules 2-3
- All algorithms from scratch
- Math proofs for key results

**Weeks 5-6**: Neural Networks
- Module 4
- Backpropagation derivation
- Train networks on MNIST, CIFAR-10

**Weeks 7-8**: Unsupervised Learning
- Module 5: K-means, PCA, EM
- Dimensionality reduction projects

**Weeks 9-10**: Deep Learning
- Module 6: CNNs, RNNs, Transformers
- Read seminal papers
- Implement architectures

## How to Use This Repo

### Study Strategy

1. **Read module**: Understand concepts
2. **Code examples**: Run and modify
3. **Practice problems**: Solve before checking solutions
4. **Implement from scratch**: Don't just use sklearn
5. **Real datasets**: Apply to UCI, Kaggle data

### Module Dependencies

**Must follow**:
1 → 2 → 3 → 4

**Can do in parallel after 4**:
5 (Unsupervised), 6 (Deep Learning)

## Daily Practice (30-60 min)

### Week 1-2: Foundations
```python
# Day 1-3: Linear algebra
- Matrix operations in NumPy
- Eigendecomposition
- SVD

# Day 4-7: Calculus & optimization
- Gradient computation
- Implement gradient descent
- Visualize convergence

# Day 8-14: Linear regression
- OLS closed-form
- Ridge & Lasso
- Cross-validation
```

### Week 3-4: Classification
```python
# Day 15-21: Logistic regression
- From scratch implementation
- Gradient descent training
- Evaluate on real data

# Day 22-28: SVM & Trees
- Understand kernel trick
- Decision tree implementation
- Random forest comparison
```

### Week 5-6: Neural Networks
```python
# Day 29-35: Feedforward networks
- Implement forward pass
- Backpropagation derivation
- Train on MNIST

# Day 36-42: PyTorch
- Learn PyTorch basics
- Build networks with nn.Module
- Train/val loops
```

## Essential Python Libraries

```bash
# Install dependencies
pip install numpy scipy matplotlib seaborn
pip install scikit-learn pandas jupyter
pip install torch torchvision

# For GPU (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Key Commands

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import torch
import torch.nn as nn
```

## Common Pitfalls

### Conceptual
1. **Confusing loss and accuracy**: Loss guides training, accuracy evaluates
2. **Overfitting**: Model memorizes training data, fails on test
3. **Underfitting**: Model too simple, high bias
4. **Data leakage**: Test data influences training

### Implementation
1. **Forgetting bias term**: Add intercept column
2. **Not standardizing**: Features on different scales
3. **Wrong shapes**: Check dimensions (n, d) vs (d, n)
4. **Numerical instability**: Use log-sum-exp for softmax

## Debugging Tips

```python
# 1. Check shapes
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"Weight shape: {w.shape}")

# 2. Monitor training
for epoch in range(epochs):
    loss = compute_loss(X, y, w)
    print(f"Epoch {epoch}: loss={loss:.4f}")

# 3. Visualize
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# 4. Sanity checks
# - Can model overfit small dataset?
# - Does loss decrease?
# - Are gradients reasonable magnitude?
```

## Project Ideas

### Beginner
1. **Housing prices**: Linear regression on UCI dataset
2. **Iris classification**: Logistic regression, 3 classes
3. **MNIST digits**: Neural network, 10 classes

### Intermediate
4. **Spam detection**: Text classification with TF-IDF
5. **Image classification**: CNN on CIFAR-10
6. **Kaggle competition**: Titanic survival prediction

### Advanced
7. **Face recognition**: Eigenfaces (PCA)
8. **Style transfer**: CNN features
9. **Sentiment analysis**: LSTM or Transformer

## Success Metrics

### After Module 1-2
- Implement linear/ridge regression from scratch
- Explain bias-variance tradeoff
- Cross-validate to select hyperparameters

### After Module 3
- Implement logistic regression, backprop
- Train SVM, interpret decision boundary
- Use precision/recall for imbalanced data

### After Module 4
- Derive backprop for 2-layer network
- Train neural network in PyTorch
- Achieve >95% on MNIST

## Study Schedule (Fast Track)

**Week 1** (Foundation + Regression):
- Mon-Tue: Module 1 foundations
- Wed-Thu: Module 2 theory
- Fri-Sun: Code OLS, Ridge, Lasso from scratch

**Week 2** (Classification):
- Mon-Tue: Logistic regression theory + code
- Wed-Thu: SVM + decision trees
- Fri-Sun: Real dataset (e.g., breast cancer classification)

**Week 3** (Neural Networks):
- Mon-Tue: Feedforward networks, backprop math
- Wed-Thu: Implement in NumPy
- Fri-Sun: PyTorch tutorial, MNIST

**Week 4** (Practice):
- Mon-Tue: CIFAR-10 CNN
- Wed-Thu: Kaggle dataset
- Fri-Sun: Review, fill gaps

## Resources Quick Links

**Textbooks** (free online):
- [ISLR](https://www.statlearning.com/) - Beginner-friendly
- [ESL](https://hastie.su.domains/ElemStatLearn/) - Advanced
- [Deep Learning Book](https://www.deeplearningbook.org/) - Goodfellow et al.

**Courses**:
- CS 189 lectures: [eecs189.org](https://eecs189.org)
- Andrew Ng ML: [Coursera](https://www.coursera.org/learn/machine-learning)
- Fast.ai: [course.fast.ai](https://course.fast.ai/)

**Datasets**:
- [UCI ML Repository](https://archive.ics.uci.edu/ml/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Papers With Code](https://paperswithcode.com/datasets)

## Math Prerequisites Review

If rusty on math, review these first:

**Linear Algebra** (1 week):
- 3Blue1Brown: Essence of Linear Algebra (YouTube)
- Gilbert Strang: Linear Algebra lectures (MIT OCW)

**Calculus** (1 week):
- Khan Academy: Multivariable Calculus
- 3Blue1Brown: Essence of Calculus

**Probability** (1 week):
- Khan Academy: Probability & Statistics
- CS 70 notes (Berkeley)

## Getting Unstuck

**Concept unclear?**
1. Re-read textbook chapter
2. Watch lecture video
3. Try worked example
4. Ask on Piazza/StackOverflow

**Code not working?**
1. Print intermediate values
2. Check shapes and types
3. Simplify to minimal example
4. Compare with sklearn implementation

**Math derivation confusing?**
1. Work through example with numbers
2. Check matrix dimensions
3. Review chain rule
4. See "Matrix Cookbook" PDF

## Next Steps After Completion

**Advanced ML**:
- Reinforcement Learning (CS 285)
- Computer Vision (CS 280)
- Natural Language Processing (CS 288)

**Research**:
- Read papers from NeurIPS, ICML, ICLR
- Implement paper from scratch
- Participate in Kaggle competitions

**Career**:
- ML Engineer: Deploy models to production
- Data Scientist: Analyze data, build predictive models
- Research Scientist: Push state-of-the-art

## Tips for Success

1. **Code first, then use libraries**: Understand before abstracting
2. **Visualize everything**: Loss curves, decision boundaries, embeddings
3. **Start simple**: Get basics working before adding complexity
4. **Real data messy**: Learn data cleaning, feature engineering
5. **Math matters**: Invest time understanding derivations
6. **Community**: Join ML Discord, Kaggle discussions

Good luck! 🚀
