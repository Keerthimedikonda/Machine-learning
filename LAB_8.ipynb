import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Load data

data = pd.read_excel("C:\\Users\\daver\\OneDrive\\Desktop\\ML\\ML-lab3dataset.xlsx")

# Process features and target
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target

# Convert categorical features to numeric if necessary
X = pd.get_dummies(X)  # One-hot encoding for categorical features

# Encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grids for each classifier
param_grids = {
    'perceptron': {
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'max_iter': [1000, 2000, 3000],
        'tol': [1e-4, 1e-3]
    },
    'mlp': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['tanh', 'relu'],
        'alpha': [0.0001, 0.001, 0.01]
    },
    'svm': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    'decision_tree': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'max_features': ['sqrt', 'log2', None],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10]
    },
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    },
    'ada_boost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    },
    'xgboost': {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3]
    },
    'naive_bayes': {
        'var_smoothing': [1e-9, 1e-8, 1e-7]
    }
}

# Define classifiers
classifiers = {
    'perceptron': (Perceptron(), param_grids['perceptron']),
    'mlp': (MLPClassifier(), param_grids['mlp']),
    'svm': (SVC(), param_grids['svm']),
    'decision_tree': (DecisionTreeClassifier(), param_grids['decision_tree']),
    'random_forest': (RandomForestClassifier(), param_grids['random_forest']),
    'ada_boost': (AdaBoostClassifier(), param_grids['ada_boost']),
    'xgboost': (XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), param_grids['xgboost']),
    'naive_bayes': (GaussianNB(), param_grids['naive_bayes']),
}
def tune_and_evaluate(model, param_grid, X_train, y_train, X_test, y_test):
    n_iter = min(10, len(param_grid)) if len(param_grid) > 0 else 1
    
    # Set cv to be the minimum of 5 or the number of samples in the smallest class
    # Convert y_train to a Pandas Series to use value_counts()
    min_samples_per_class = pd.Series(y_train).value_counts().min()
    # Set cv to be 2 if min_samples_per_class is less than or equal to 2, otherwise set it to 5
    # If min_samples_per_class is 1, set cv to LeaveOneOut()
    if min_samples_per_class == 1:
        from sklearn.model_selection import LeaveOneOut
        cv = LeaveOneOut()
    else:
        cv = 2 if min_samples_per_class <= 2 else 5
    
    search = RandomizedSearchCV(model, param_grid, n_iter=n_iter, cv=cv, random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    return {
        'best_params': search.best_params_,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
    }
results = []

for name, (model, params) in classifiers.items():
    metrics = tune_and_evaluate(model, params, X_train, y_train, X_test, y_test)
    metrics['Classifier'] = name
    results.append(metrics)

# Print performance metrics
print("Performance Metrics:")
for result in results:
    print(f"Classifier: {result['Classifier']}")
    print(f"  Accuracy: {result['accuracy']:.4f}")
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  Recall: {result['recall']:.4f}")
    print(f"  F1 Score: {result['f1_score']:.4f}")
    print()

# Print hyperparameters
print("Hyperparameters:")
for result in results:
    print(f"Classifier: {result['Classifier']}")
    print(f"  Best Hyperparameters: {result['best_params']}")
    print()
