import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# PART 1: Manual Naive Bayes Implementation
# ==========================================

class CustomNaiveBayes:
    """
    A Naive Bayes classifier implemented from scratch to demonstrate
    understanding of conditional probability and Laplace smoothing.
    """
    def fit(self, X, y):
        """
        Fits the model by storing training data. 
        In Naive Bayes, 'fitting' is mostly about analyzing the training set.
        """
        self.X_train = X
        self.y_train = y
        self.classes = np.unique(y)
        return self

    def _calculate_prior(self, target_class):
        """
        Calculates P(Class): The probability of a specific class occurring.
        """
        return np.mean(self.y_train == target_class)

    def _calculate_likelihood(self, feature_index, feature_val, target_class):
        """
        Calculates P(Feature | Class) using Laplace Smoothing.
        
        Laplace Smoothing Formula:
        (Count(feature_val in class) + 1) / (Count(total in class) + Count(unique_features))
        
        The '+1' prevents zero-division errors if a feature value wasn't seen in training.
        """
        # Filter training data for the specific class
        class_subset = self.X_train[self.y_train == target_class]
        
        numerator = np.sum(class_subset[:, feature_index] == feature_val) + 1
        denominator = class_subset.shape[0] + len(np.unique(self.X_train[:, feature_index]))
        
        return numerator / denominator

    def predict(self, X_test):
        """
        Predicts class labels for new data.
        """
        predictions = []
        
        for sample in X_test:
            posteriors = []
            
            # Calculate posterior probability for each class
            for target_class in self.classes:
                # Start with the Prior P(Class)
                posterior = self._calculate_prior(target_class)
                
                # Multiply by Likelihoods P(Feature | Class) for all features
                for feature_index in range(len(sample)):
                    posterior *= self._calculate_likelihood(
                        feature_index, 
                        sample[feature_index], 
                        target_class
                    )
                posteriors.append(posterior)
            
            # Choose the class with the highest probability
            predictions.append(self.classes[np.argmax(posteriors)])
            
        return np.array(predictions)

# --- Execution: Mushroom Classification ---
def run_mushroom_analysis():
    print("--- Running Mushroom Classification (From Scratch) ---")
    try:
        # Load Data
        df = pd.read_csv('mushrooms.csv')
        
        # Prepare Features and Target
        X = df.drop(['class'], axis=1).values
        y = df['class'].values

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.95, random_state=42)

        # Train Custom Model
        model = CustomNaiveBayes()
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Evaluate
        acc = np.mean(y_test == y_pred)
        print(f"Custom Naive Bayes Accuracy: {acc:.4f}")
        
    except FileNotFoundError:
        print("Error: 'mushrooms.csv' not found. Skipping...")

# ==========================================
# PART 2: Stress Level Analysis (Using Libraries)
# ==========================================

def run_stress_analysis():
    print("\n--- Running Stress Level Analysis (Scikit-Learn) ---")
    try:
        # Load Data (Renamed 'da' to 'stress_df' for clarity)
        stress_df = pd.read_csv('StressLevelDataset.csv')

        # Visualization (Optional: usually done in notebooks, but kept for script completeness)
        # plt.figure(figsize=(10, 6))
        # sns.histplot(stress_df, x='stress_level', y='peer_pressure')
        # plt.title("Peer Pressure vs Stress Level")
        # plt.show()

        X = stress_df.drop(['stress_level'], axis=1).values
        y = stress_df['stress_level'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.95, random_state=42)

        # 1. Scikit-Learn Categorical Naive Bayes
        nb_model = CategoricalNB()
        nb_model.fit(X_train, y_train)
        nb_pred = nb_model.predict(X_test)
        
        print("CategoricalNB Accuracy:", accuracy_score(y_test, nb_pred))

        # 2. Logistic Regression Comparison
        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        
        print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
        
        # detailed report
        print("\nClassification Report (Logistic Regression):")
        print(classification_report(y_test, lr_pred))

    except FileNotFoundError:
        print("Error: 'StressLevelDataset.csv' not found. Skipping...")

if __name__ == "__main__":
    run_mushroom_analysis()
    run_stress_analysis()