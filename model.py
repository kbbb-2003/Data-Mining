# Install necessary libraries

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import shap

# Load the dataset
data = pd.read_csv('dataset.csv')  # Replace with the actual path to your dataset

# Preprocess data: handle missing values, outliers, etc.
data = data.dropna()  # Example of handling missing values, adjust as needed

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.95)  # Retain 95% of variance
principal_components = pca.fit_transform(data.drop(columns=['target']))  # Assuming 'target' is the label column

# K-Means clustering
kmeans = KMeans(n_clusters=2)  # Assuming two clusters: hyperinsulinemic and non-hyperinsulinemic
clusters = kmeans.fit_predict(principal_components)
data['cluster'] = clusters

# Split data into training and test sets
X = data.drop(columns=['target'])
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
rf = RandomForestClassifier(random_state=42)
xgb = XGBClassifier(random_state=42)
lgbm = LGBMClassifier(random_state=42)

# Train models
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)
lgbm.fit(X_train, y_train)

# Evaluate model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    return accuracy, precision, recall, f1, mcc

rf_metrics = evaluate_model(rf, X_test, y_test)
xgb_metrics = evaluate_model(xgb, X_test, y_test)
lgbm_metrics = evaluate_model(lgbm, X_test, y_test)

print("Random Forest metrics:", rf_metrics)
print("XGBoost metrics:", xgb_metrics)
print("LightGBM metrics:", lgbm_metrics)

# SHAP for model explanation
explainer = shap.TreeExplainer(rf)  # Using Random Forest as an example
shap_values = explainer.shap_values(X_test)

# SHAP summary plot
shap.summary_plot(shap_values, X_test, feature_names=X.columns)

# Feature importance
shap.summary_plot(shap_values[1], X_test, plot_type="bar")  # For binary classification, use shap_values[1]
