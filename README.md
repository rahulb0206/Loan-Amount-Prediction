# Loan Amount Prediction using TensorFlow
This project focuses on predicting loan default status by analyzing various factors such as age, income, home ownership, loan intent, and credit history. Accurate loan default predictions are crucial for financial institutions as they assist in assessing risk, making informed lending decisions, and minimizing potential losses. The deployment of the model in TensorFlow Lite format ensures it is optimized for use on mobile and edge devices, enabling real-time decision-making in diverse environments. The integration of this model into a Flutter application, developed using Android Studio, further enhances accessibility by providing a user-friendly interface for real-time predictions on both mobile and web platforms.

# Model Creation and Training
In this phase, the dataset credit_risk.csv is loaded and preprocessed to handle missing values and encode categorical features. The neural network architecture is designed using TensorFlow's Keras API, featuring multiple dense layers with ReLU activation functions and a final output layer with a sigmoid activation for binary classification. The model is then trained using the Adam optimizer and binary cross-entropy loss function. Early stopping and model checkpointing techniques are employed to prevent overfitting and ensure that the best version of the model is saved. TensorBoard is integrated to monitor the training process in real-time. After training, the model is converted to TensorFlow Lite format (.tflite), making it suitable for deployment on mobile and edge devices.

# Model Prediction
The TensorFlow Lite model (.tflite) is utilized in a Flutter application. The code for model prediction is provided in a separate script (prediction.py), which handles the loading of the TensorFlow Lite model, input of data, and generation of predictions. The Flutter application, developed using Android Studio, integrates the TensorFlow Lite model using the tflite_flutter package. This allows for efficient inference directly on mobile devices. Users can input details such as age, income, home ownership status, loan specifics, and credit history through the app’s interface. The app processes these inputs, runs the model to predict loan default status, and displays the results in a user-friendly format. Additionally, the project includes an output video showcasing the Flutter Web interface, demonstrating how the predictions are displayed in a web-based environment.

# Python Notebook
The entire process of model creation, training, and conversion to TensorFlow Lite format is coded and executable in a Jupyter notebook (project.ipynb). This notebook provides a step-by-step guide for loading and preprocessing the dataset, designing and training the neural network, converting the trained model to TensorFlow Lite format, and predicting the loan amount. The notebook is an alternative way to run and explore the project, allowing users to directly execute the code and observe the results interactively.

<img width="1403" alt="ipynb" src="https://github.com/user-attachments/assets/4283c068-ecae-4ac7-98d6-ab47103cc079">


# Project Files
The project consists of the following components:

- project.ipynb: Jupyter notebook for loading and preprocessing the dataset, building and training the neural network, converting the model to TensorFlow Lite format, and predicting the loan amount. This notebook enables users to execute the entire model development process interactively.
- model.py: Script for defining and training the neural network model.
- prediction.py: Script for loading the TensorFlow Lite model, accepting user input, performing predictions, and displaying results.
- Flutter Application: Developed using Android Studio, this application integrates the TensorFlow Lite model via the tflite_flutter package, allowing for real-time predictions on mobile devices. The output video demonstrates how the Flutter Web interface showcases predictions.

# Requirements
- Python 3 or above
- TensorFlow 2 or above
- NumPy
- Pandas
- Android Studio
- Jupyter Notebook (for running .ipynb file)

# Output Video

https://github.com/user-attachments/assets/e1652313-a3e4-4a65-a995-31e4786746f6


