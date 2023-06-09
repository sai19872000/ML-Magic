import pandas as pd
import numpy as np
from sklearn.datasets import make_classification



# Create a synthetic dataset
def create_dataset(n_samples=1000, n_features=10, n_informative=2, 
                   n_redundant=2, n_repeated=0, n_classes=2, 
                   n_clusters_per_class=2, weights=None, 
                   flip_y=0.01, class_sep=1.0, hypercube=True, 
                   shift=0.0, scale=1.0, shuffle=True, random_state=8):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                               n_informative=n_informative, n_redundant=n_redundant, 
                               n_repeated=n_repeated, n_classes=n_classes, 
                               n_clusters_per_class=n_clusters_per_class, weights=weights, 
                               flip_y=flip_y, class_sep=class_sep, hypercube=hypercube, 
                               shift=shift, scale=scale, shuffle=shuffle, random_state=random_state)

    df = pd.DataFrame(X)
    df['target'] = y
    # Introduce some NaN values randomly for data cleaning tutorial
    for col in df.columns:
        df.loc[df.sample(frac=0.1).index, col] = np.nan

    # Introduce some categorical data for one hot encoding
    df['Category'] = pd.Series(np.random.choice(['A', 'B', 'C', 'D'], size=len(df)))
    df['Category'] = df['Category'].astype('category')
    return df

# python function to calculate capture rate at a given percentage
def capture_rate(y_true, y_pred_proba, percentage):
    """
    This function calculates the capture rate at a given percentage
    """
    # sort the probabilities in descending order
    y_pred_proba_sorted = np.sort(y_pred_proba)[::-1]

    # calculate the number of observations we need to select
    number_obs = int(np.ceil(percentage/100 * len(y_true)))

    # select the observations
    selected_y_pred_proba = y_pred_proba_sorted[:number_obs]

    # get the threshold
    threshold = selected_y_pred_proba[-1]

    # get the predictions
    y_pred = [1 if proba >= threshold else 0 for proba in y_pred_proba]

    # calculate the capture rate
    capture_rate = sum(y_pred)/sum(y_true)

    return capture_rate