import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.regularizers import l2

sys.path.append("../")
from helper.config import *
from helper.utility_functions import *
import model_functions
import nn_model

# Path to save or load the model
X, y = load_data(image_data_path)
split_dataset(X, y, test_size=0.3, test_split=0.5)

# Debugging lines
print("Shape of X:", np.shape(X))
print("Shape of y:", np.shape(y))

# Preprocess the images
datagen = preprocess_images(X)

model = load_model(model_path)

hidden_layers = [4, 16, 128, 32]

if model is None:
    parameters, costs = L_layer_model(X_train, y, hidden_layers, 0.0075, 3000, False)
    print("New model created.")

# Fit the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=200,
                    validation_data=(X_cv, y_cv))

# Save the model after training
save_model(model, model_path)

# Plotting
plot_metrics(history)

predictions = predict(X_test)
y_pred = np.argmax(predictions, axis=1)
num_errors = np.sum(y_pred != y_test)

print_model_summary(model)
evaluate_model(model, X_test, y_test)