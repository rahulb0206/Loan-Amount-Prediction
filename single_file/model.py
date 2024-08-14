import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os
import warnings
warnings.filterwarnings("ignore")


# Load the dataset
credit = pd.read_csv('credit_risk.csv')

# Drop columns with missing values
credit.dropna(axis=1, inplace=True)

# Create LabelEncoder object
label_encoder = LabelEncoder()

# Apply label encoding to each categorical column
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

for col in categorical_cols:
    credit[col] = label_encoder.fit_transform(credit[col])

# Split data into features (X) and target variable (y)
X = credit.drop('loan_status', axis=1)
y = credit['loan_status']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network architecture
NNmodel = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),  # Additional hidden layer
    Dense(8, activation='relu'),   # Additional hidden layer
    Dense(1, activation='sigmoid')
])

# Compile the model
NNmodel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define TensorBoard callback with an absolute log directory path
log_dir = os.path.join(r"R:\Running project\Android Credit risk analysis custome\Single_file", "logs")  # Use 'r' before the string to treat it as a raw string
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Define other callbacks (EarlyStopping and ModelCheckpoint)
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
    tf.keras.callbacks.ModelCheckpoint(filepath='best_model.keras', monitor='val_accuracy', save_best_only=True)
]

# Train the model with TensorBoard callback
history = NNmodel.fit(X_train, Y_train, epochs=50, batch_size=32,
                      validation_data=(X_test, Y_test),
                      callbacks=[tensorboard_callback] + callbacks)

# Evaluate the model on the test set
test_loss, test_accuracy = NNmodel.evaluate(X_test, Y_test)
print(f"Test Accuracy: {test_accuracy}")

new_model = tf.keras.models.load_model('best_model.keras')

# Show the model architecture
new_model.summary()

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
