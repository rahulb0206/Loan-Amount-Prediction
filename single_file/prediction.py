import numpy as np
import tensorflow as tf
import os
import warnings
warnings.filterwarnings("ignore")

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def get_input_data():
    person_age = int(input("Enter person's age: "))
    person_income = int(input("Enter person's income: "))
    person_home_ownership = int(input("Enter person's home ownership (0 - Rented, 1 - Mortgage, 2 - Own, 3 - Other): "))
    loan_intent = int(input("Enter loan intent (1 - Debt Consolidation, 2 - Home Improvement, 3 - Medical Expenses, 4 - Other, 5 - Personal): "))
    loan_grade = int(input("Enter loan grade (1 - A, 2 - B, 3 - C, 4 - D, 5 - E, 6 - F, 7 - G): "))
    loan_amnt = int(input("Enter loan amount: "))
    loan_percent_income = float(input("Enter loan percent of income: "))
    cb_person_default_on_file = int(input("Enter person's default history (0 - No, 1 - Yes): "))
    cb_person_cred_hist_length = int(input("Enter person's credit history length (in years): "))

    # Prepare input data as a numpy array
    input_data = np.array([[person_age, person_income, person_home_ownership, loan_intent, loan_grade,
                            loan_amnt, loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length]],
                          dtype=np.float32)

    return input_data

def predict_loan_status(input_data):
    # Ensure the input shape matches the model's input tensor shape
    input_shape = input_details[0]['shape']
    input_data = np.reshape(input_data, input_shape)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Since it's a binary classification (assuming sigmoid output), round to get binary prediction
    binary_prediction = 1 if output_data[0][0] >= 0.5 else 0

    return binary_prediction

while True:
    # Get input data from user
    input_data = get_input_data()

    # Predict loan status
    prediction = predict_loan_status(input_data)

    # Print prediction
    print(f"Predicted loan status (0 - Not Defaulted, 1 - Defaulted): {prediction}")

    # Ask if user wants to continue
    continue_prompt = input("Do you want to predict again? (yes/no): ").lower()
    if continue_prompt != 'yes':
        break
