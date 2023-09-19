import sys
import pickle
import numpy as np

sys.path.append("../")
from helper.config import *
from helper.utility_functions import *
import model_functions
import nn_model

# Path to save or load the model
X, y = load_data(image_data_path)
print(X[:2])
print(X[0].shape)
print(y[0].shape)
print("Shape of X before split:", np.shape(X))

X_train, X_test, X_cv, y_train, y_test, y_cv = split_dataset(X, y, test_size=0.3, test_split=0.5)

# Debugging lines
print("Shape of X:", np.shape(X))
print("Shape of y:", np.shape(y))
print("Shape of X_train:", np.shape(X_train))
print("Shape of X_test:", np.shape(X_test))
print("Shape of X_cv:", np.shape(X_cv))

# Preprocess the images
print(f"Shape of X_train before preprocess_images: {X_train.shape}")
datagen = preprocess_images(X_train)

model = load_model(model_path)

hidden_layers = [4, 16, 128, 32]

# if model is None:
parameters, history = L_layer_model(X_train, y, X_cv, y_cv, hidden_layers, 0.0075, 3000, False, datagen)
print("New model created.")

predictions = predict_multi(X_test, parameters)
evaluate_model(predictions, y_test)

# Save the model after training
save_model(parameters, model_path)

# Plotting
plot_metrics(history)

# Print model summary
print_model_summary(hidden_layers)