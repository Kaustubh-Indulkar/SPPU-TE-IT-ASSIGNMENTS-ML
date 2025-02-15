# SPPU-ASSIGNMENTS-ML

# Machine Learning Assignments

This repository contains implementations of various machine learning techniques, covering regression, classification, model performance improvement, clustering, association rule learning, and neural networks.

## Table of Contents

1. [Regression](#regression)
2. [Classification](#classification)
3. [Improving Classifier Performance](#improving-classifier-performance)
4. [Clustering](#clustering)
5. [Association Rule Learning](#association-rule-learning)
6. [Multilayer Neural Network](#multilayer-neural-network)

## Regression

* **Dataset:** [Temperatures of India](https://www.kaggle.com/venky73/temperatures-of-india?select=temperatures.csv)
* **Task:** Apply linear regression to predict month-wise temperatures.
* **Metrics:** MSE, MAE, R-squared
* **Visualizations:** Simple regression model visualization.

## Classification

* **Dataset:** [Graduate Admissions](https://www.kaggle.com/mohansacharya/graduate-admissions)
* **Task:** Build a decision tree classifier to predict university admission based on GRE and academic scores.
* **Preprocessing:** Label encoding, data transformation (if necessary).
* **Steps:**
    * Data preparation (train-test split)
    * Apply decision tree algorithm
    * Evaluate model

## Improving Classifier Performance

* **Dataset:** [SMS Spam Collection](http://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
* **Task:** Implement an SMS spam filtering system using a probabilistic approach (Naive Bayes/Bayesian Network).
* **Features:** Message length, word count, unique keywords, etc.
* **Steps:**
    * Data preprocessing
    * Train-test split
    * Apply at least two machine learning algorithms and evaluate
    * Cross-validation and evaluation
    * Hyperparameter tuning and evaluation

## Clustering

* **Dataset:** [Mall Customers](https://www.kaggle.com/shwetabh123/mall-customers)
* **Task:** Apply clustering algorithms (based on spending score) to segment customers into profitable groups.
* **Algorithms:** At least two clustering algorithms.
* **Steps:**
    * Data preprocessing
    * Train-test split (if applicable)
    * Apply clustering algorithms
    * Evaluate model
    * Cross-validation and evaluation (if applicable)

## Association Rule Learning

* **Dataset:** [Market Basket Optimization](https://www.kaggle.com/hemanthkumar05/market-basket-optimization)
* **Task:** Find association rules between items in retail transactions using the Apriori algorithm.
* **Steps:**
    * Data preprocessing
    * Generate transaction list
    * Train Apriori algorithm
    * Visualize rules
    * Explore rules with varying minimum confidence.

## Multilayer Neural Network

* **Dataset:** [Pima Indians Diabetes](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)
* **Task:** Build a multilayer neural network to predict diabetes.
* **Architecture:** Two hidden layers with ReLU activation, sigmoid output layer.
* **Steps:**
    * Load and define the model (Keras)
    * Compile and fit the model
    * Evaluate performance with different epochs and batch sizes
    * Evaluate with different activation functions
    * Visualize the model (using ANN Visualizer).

## Requirements

List all required libraries and their versions (e.g., `pandas`, `numpy`, `scikit-learn`, `tensorflow`, `keras`, `mlxtend`).  

