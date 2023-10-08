import sys
import pickle
import numpy as np

sys.path.append("../")
from helper.config import *
from helper.utility_functions import *
from nn_model import L_layer_model

X, y = load_data(image_data_path)
print("X.shape", X.shape)
print("y.shape", y.shape)
y = y.reshape(y.shape[0], 1)
print("y.shape", y.shape)

y = one_hot_encode(y)
# y = y.T
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
X_flatten = X.reshape(X.shape[0], -1).T
X_standardized = X_flatten/255

print(X_train_standardized.shape)
print(X_cv_standardized.shape)
print(X_test_standardized.shape)
print(y_train.shape)
print(y_cv.shape)
print(type(y_cv))
print(y_test.shape)

model_parameters = load_model(model_path)

hidden_layers = [52441, 50, 30, 15, 2]

if model_parameters is None:
    parameters, history = L_layer_model(X_train_standardized, y_train, hidden_layers, None, None, 0.001, 1000, True)
    his_true = True
    print("New model created.")
else:
    parameters = model_parameters
    his_true = False

y_test = np.argmax(y_test, axis=1)
y_cv = np.argmax(y_cv, axis=1)
y_full = np.argmax(y, axis=1)
print("X TEST:")
predict_and_evaluate(X_test_standardized, y_test, parameters)
print("X CV:")
predict_and_evaluate(X_cv_standardized, y_cv, parameters)
print("X:")
predict_and_evaluate(X_standardized, y, parameters)

# Save the model after training
save_model(parameters, model_path)

# Plotting
if his_true:
    plot_metrics(history)
else:
    print("No history found.")

# Print model summary
print_model_summary(hidden_layers)