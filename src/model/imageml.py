import sys
import pickle
import numpy as np

sys.path.append("../")
from helper.config import *
from helper.utility_functions import *
import model_functions
from nn_model import L_layer_model

# Path to save or load the model
X, y = load_data(image_data_path)
print("X.shape", X.shape)
print("y.shape", y.shape)
y = y.reshape(y.shape[0], 1)
print("y.shape", y.shape)

y = one_hot_encode(y, num_classes=3)
y = y.T
print("y.shape", y.shape)

X_train, X_test, X_cv, y_train, y_test, y_cv = split_dataset(X, y, test_size=0.4, test_split=0.5)


# Preprocess the images
# datagen = preprocess_images(X_train)
X_train_flatten = X_train.reshape(X_train.shape[0], -1).T
X_train_standardized = X_train_flatten/255
X_cv_flatten = X_cv.reshape(X_cv.shape[0], -1).T
X_cv_standardized = X_cv_flatten/255
X_test_flatten = X_test.reshape(X_test.shape[0], -1).T
X_test_standardized = X_test_flatten/255
#y_train = y_train.reshape(y_train.shape[0], 1)
# y_train_reshaped = y_train.reshape(1, -1)
print(X_train_standardized.shape)
print(X_cv_standardized.shape)
print(X_test_standardized.shape)
print(y_train.shape)
# print(y_train_reshaped.shape)

model = load_model(model_path)

hidden_layers = [52441, 31, 7, 5, 3]

if model is None:
    parameters, history = L_layer_model(X_train_standardized, y_train, hidden_layers, None, None, 0.0075, 3000, True)
    print("New model created.")

predictions = predict_results(X_test_standardized, parameters)
evaluate_model(predictions, y_test)

# Save the model after training
save_model(parameters, model_path)

# Plotting
plot_metrics(history)

# Print model summary
print_model_summary(hidden_layers)