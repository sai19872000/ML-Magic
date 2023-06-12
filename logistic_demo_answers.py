from utils import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn import utils

# Create a synthetic dataset
df = create_dataset(n_samples=1000, n_features=10, n_informative=2)
df.columns = df.columns.astype(str) # convert column names to string

# saving the dataframe
df.to_csv('synthetic_data.csv', index=False)

# Get some basic info about your DataFrame
print(df.info())

# get stats on numeric columns
print(df.describe())

# Handle missing data
# Replace NaNs with the mean of the column (for numerical columns)
for column in df.select_dtypes(include=[np.number]).columns:
    df[column] = df[column].fillna(df[column].mean())

# Replace NaNs with the mode of the column (for categorical columns)
for column in df.select_dtypes(include=[object]).columns:
    df[column] = df[column].fillna(df[column].mode()[0])

# One hot encode categorical variables
df = pd.get_dummies(df)

# Get some basic info about your DataFrame
print(df.info())

# get stats on numeric columns
print(df.describe())

# Scaling numerical features to have mean = 0 and variance = 1
scaler = StandardScaler()
for column in df.select_dtypes(include=[np.number]).columns:
    df[column] = scaler.fit_transform(df[[column]])

# Split dataset into features (X) and target (y)
X = df.drop('target', axis=1) # replace 'target' with your target column
y = df['target']


#convert y values to categorical values
lab = preprocessing.LabelEncoder()
y = lab.fit_transform(y)

# convert x into numpy array
X_asa = np.asarray(X).astype('float32')

# Split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X_asa, y, test_size=0.2, random_state=42)

#print sample of y_train
print(set(y_train))

# Initialize and fit Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")

# calculate model probabilities
y_prob = model.predict_proba(X_test)
print(y_prob.shape)
# calculate roc_auc_score
print(f"ROC AUC Score: {metrics.roc_auc_score(y_test, y_prob, multi_class='ovr',average=None)}")