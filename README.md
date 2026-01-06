# Comparative Analysis of Naive Bayes Algorithms

## Overview
This project explores the mathematical underpinnings of the Naive Bayes classification algorithm. It features a dual approach:
1.  **Manual Implementation:** A custom Python class built from scratch to demonstrate the calculation of Priors, Likelihoods, and Laplace Smoothing without relying on external libraries.
2.  **Library Optimization:** A comparative analysis using Scikit-Learn's `CategoricalNB` and `LogisticRegression` to benchmark performance.

## Datasets
* **Mushroom Dataset:** Used for the manual implementation to classify samples as Edible ('e') or Poisonous ('p').
* **Stress Level Dataset:** Used for the Scikit-Learn comparison to predict stress levels based on psychological and environmental factors.

## Implementation Details

### Custom Naive Bayes (From Scratch)
Unlike standard library calls, I implemented the algorithm mathematically:
* **Prior Probability:** Calculated based on class frequency in the training set.
* **Likelihood with Laplace Smoothing:** Implemented the formula `(count + 1) / (total + unique)` to handle zero-frequency issues where a feature in the test set was not present in the training set.

### Performance
* **Custom Model Accuracy:** ~96.8% (Mushroom Dataset)
* **Scikit-Learn CategoricalNB:** ~[Insert Score]% (Stress Dataset)

## Technologies
* Python
* NumPy (for matrix operations)
* Pandas (for data manipulation)
* Scikit-Learn (for benchmarking)
* Seaborn/Matplotlib (for EDA)

## How to Run
```bash
python naive_bayes_project.py
