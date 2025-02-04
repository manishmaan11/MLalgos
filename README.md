# Machine Learning Algorithms from Scratch

This repository contains implementations of various **Machine Learning (ML) algorithms** developed **from scratch** using only basic Python libraries (without relying on built-in ML packages such as `scikit-learn`, `TensorFlow`, or `PyTorch`). The goal of this repository is to provide a deeper understanding of how different ML algorithms work by implementing them manually.

## Table of Contents

- [Introduction](#introduction)
- [Algorithms Implemented](#algorithms-implemented)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The purpose of this repository is to help you better understand the mechanics of machine learning algorithms by walking through their implementation from scratch. By building these models without relying on high-level libraries, we get a clearer view of the underlying math and logic that drives machine learning.

These implementations serve as a learning resource for students, practitioners, and anyone interested in exploring the foundations of ML. The code is written in Python, with no external libraries beyond `NumPy` for numerical computations.

## Algorithms Implemented

Below is a list of machine learning algorithms currently implemented in this repository:

### Supervised Learning
- **Linear Regression**  
  Simple implementation of linear regression to predict continuous values.
  
- **Logistic Regression**  
  Binary classification using logistic regression with gradient descent.

- **K-Nearest Neighbors (KNN)**  
  A simple KNN algorithm for classification based on proximity to nearest data points.

- **Decision Trees**  
  Recursive binary tree structure used for classification and regression tasks.

- **Support Vector Machine (SVM)**  
  A classifier that works by finding the optimal hyperplane that separates data points.

- **Naive Bayes Classifier**  
  Probabilistic classifier based on Bayes’ theorem with strong (naive) independence assumptions.

### Unsupervised Learning
- **K-Means Clustering**  
  Unsupervised clustering algorithm that partitions data into K clusters.

- **Hierarchical Clustering**  
  Agglomerative clustering method that builds a hierarchy of clusters.

### Neural Networks & Deep Learning
- **Single-layer Perceptron (SLP)**  
  Basic implementation of a neural network with one hidden layer for binary classification tasks.

### Reinforcement Learning
- **Q-Learning**  
  Basic implementation of the Q-learning algorithm for reinforcement learning.

## Getting Started

To get started, follow the steps below:

### Prerequisites

- Python 3.x
- `NumPy` (for numerical operations)

You can install NumPy by running the following command:

```bash
pip install numpy
