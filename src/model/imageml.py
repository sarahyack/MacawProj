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

X_train, X_test, X_cv, y_train, y_test, y_cv = split_dataset(X, y, test_size=0.3, test_split=0.5)



# Preprocess the images
# datagen = preprocess_images(X_train)
X_train_flatten = X_train.reshape(X_train.shape[0], -1).T
X_train_standardized = X_train_flatten/255
print(X_train_standardized.shape)


# model = load_model(model_path)

hidden_layers = [12288, 20, 7, 5, 1]

# if model is None:
parameters, history = L_layer_model(X_train_standardized, y_train, hidden_layers, X_cv, y_cv, 0.0075, 3000, False)
print("New model created.")

predictions = predict_multi(X_test, parameters)
evaluate_model(predictions, y_test)

# Save the model after training
save_model(parameters, model_path)

# Plotting
plot_metrics(history)

# Print model summary
print_model_summary(hidden_layers)