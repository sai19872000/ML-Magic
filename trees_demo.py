#python program to build a xgboost model on custom dataset
import pandas as pd
import numpy as np
from utils import *
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import preprocessing
from sklearn import metrics
from matplotlib import pyplot

# load the dataset
df = pd.read_csv('synthetic_data.csv')

# print the first 5 rows of the dataframe
print(df.head())

# Split dataset into features (X) and target (y)
X = df.drop('target', axis=1) # replace 'target' with your target column
y = df['target']

#convert y values to categorical values
lab = preprocessing.LabelEncoder()
y = lab.fit_transform(y)

# index the categorical columns
categorical_columns = [c for c in X.columns if X[c].dtype.name == 'object']

# convert categorical columns to numeric
for col in categorical_columns:
    X[col] = pd.Categorical(X[col]).codes

# built xgboost model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model.fit(X_train, y_train)

# save the model
xgb_model.save_model('xgb_model.json')

# load the model
loaded_model = xgb.XGBClassifier()
loaded_model.load_model('xgb_model.json')

# make predictions on test data
y_pred = loaded_model.predict(X_test)

# evaluate the model
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")

# calculate model probabilities
y_pred_proba = loaded_model.predict_proba(X_test)

# print the first 5 rows of y_pred_proba
print(y_pred_proba[:5])

# calculate the auc score
print(f"ROC AUC Score: {metrics.roc_auc_score(y_test, y_pred_proba, multi_class='ovr',average=None)}")

# calculate feature importance
print(loaded_model.feature_importances_)

# plot feature importance using matplotlib
pyplot.bar(range(len(loaded_model.feature_importances_)), loaded_model.feature_importances_)

# add labels to the plot
pyplot.xticks(range(len(loaded_model.feature_importances_)), X.columns, rotation='vertical')

# save the plot to a file
pyplot.savefig('feature_importance.png')

# one hot encode y_test
y_test = pd.get_dummies(y_test,dtype=int)

# convert y_test to numpy array
y_test = np.asarray(y_test)

# calculate the capture rate at 10%
print(f"Capture Rate at 10%: {capture_rate(y_test[:,1], y_pred_proba[:,1], 10)}")

