import numpy as np
from sklearn import preprocessing

def extract_target_variable(data):
    """
    Extracts the target variable 'Class' from the dataframe as a NumPy array.
    """
    return data['Class'].to_numpy()

def standardize_features(X):
    """
    Standardizes the feature matrix using StandardScaler.
    """
    transform = preprocessing.StandardScaler()
    return transform.fit_transform(X)
