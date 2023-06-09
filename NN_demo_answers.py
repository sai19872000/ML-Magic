from utils import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn import utils

# for neural network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# for plotting
from matplotlib import pyplot

# load the dataset
df = pd.read_csv('synthetic_data.csv')

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

# Define a neural network model for 3 classes
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Convert y_train into one-hot encoded array
y_train = np.eye(3)[y_train]
y_test = np.eye(3)[y_test]


# Fit the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy}')

# calculate model probabilities
y_prob = model.predict(X_test)

# calculate roc_auc_score

roc_auc_score = metrics.roc_auc_score(y_test, y_prob, multi_class='ovr',average=None)
print(f"ROC AUC Score: {roc_auc_score}")

# plot the log loss for each epoch
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()

# save the plot
pyplot.savefig('NN_log_loss.png')

# close the plot
pyplot.close()

# plot the accuracy for each epoch
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()

# save the plot
pyplot.savefig('NN_accuracy.png')
pyplot.close()
