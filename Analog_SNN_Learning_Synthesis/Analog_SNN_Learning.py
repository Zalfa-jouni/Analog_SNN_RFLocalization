# This script sets up the environment for training and testing the analog-based neural network using TensorFlow and Keras.

# -- Import Libraries --

# Core libraries
import tensorflow as tf
import numpy as np
import random
import os
import pandas as pd
from scipy.interpolate import interp1d


# Data processing and model selection
import scipy.io
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Keras model and layers
import keras
from keras.models import Model
from keras.layers import Dense, Input, Dropout, Concatenate

# Optimizers
from keras.optimizers import Adam, SGD

# Visualization
import matplotlib
import matplotlib.pyplot as plt

# -- Set the Seed --

# Function Definitions
def set_seed(seed):
    """
    Sets the seed for various random number generators to ensure reproducibility.

    """
    tf.random.set_seed(seed)       # Set seed for TensorFlow
    np.random.seed(seed)           # Set seed for NumPy
    random.seed(seed)              # Set seed for Python's built-in random module
    os.environ['PYTHONHASHSEED'] = str(seed)  # Set seed for Python hash seeding
set_seed(40)

# -- Data Import and Preprocessing --

# Load the data from a MATLAB file
mat = scipy.io.loadmat('Simulated_Dataset.mat')
features = mat['features_frequencies']  # Extracting features powers from the dataset
labels = mat['labels']      # Extracting labels from the dataset

# Configuration Parameters
num_angles = 360 # Number of angles 
num_regions = 4  # Number of regions
resolution = 10  # Resolution of output angle

# Calculate the number of classes for angles and regions based on the resolution
num_angle_classes = num_angles // (num_regions * resolution)  
num_region_classes = num_regions

# Extract region and angle labels
# The first column for regions, the second column for angles
labels_regions = labels[:, 0].astype(int)     # Extracting region labels
labels_angles = labels[:, 1].astype(int)      # Extracting angle labels
#labels_angles_adjusted = labels_angles - 1   # Adjusting angle labels for zero-indexing (depends on the Matlab code)
labels_angles_adjusted = labels_angles 

# Convert labels to categorical format before splitting
labels_regions_categorical = tf.keras.utils.to_categorical(labels_regions, num_classes=num_region_classes)
labels_angles_categorical = tf.keras.utils.to_categorical(labels_angles_adjusted, num_classes=num_angle_classes)

# Split the data into training, validation, and testing sets
features_train, features_test, labels_regions_train, labels_regions_test, \
labels_angles_train, labels_angles_test = train_test_split(features, labels_regions_categorical, labels_angles_categorical, test_size=0.2)

features_train, features_val, labels_regions_train, labels_regions_val, \
labels_angles_train, labels_angles_val = train_test_split(features_train, labels_regions_train, labels_angles_train, test_size=0.25)

# Normalize the feature data
scaler = StandardScaler()
scaler.fit(features_train)      # Fit the scaler only on training data
features_train = scaler.transform(features_train)  # Apply transformation to training data
features_val = scaler.transform(features_val)      # Apply transformation to validation data
features_test = scaler.transform(features_test)    # Apply transformation to test data

# -- Set the Used Activation Function --
def custom_sigmoid(x):
    """
    Custom sigmoid activation function.

    Args:
    x (tensor): Input tensor to the activation function.

    This function applies a modified sigmoid operation on the input tensor. 
    It's characterized by parameters a, b, c, and d that adjust the curve's shape.

    The output is further scaled to ensure that its range is between 0 and 1.

    Returns:
    Tensor: The transformed tensor after applying the custom sigmoid and scaling.
    """

    # Parameters defining the custom sigmoid curve
    a = 0.99795908  # Scale factor for the sigmoid's height
    b = 1.33054425  # Controls the steepness of the sigmoid curve
    c = 0.63399893  # Adjusts the x-axis placement of the sigmoid curve
    d = -0.03544812 # Vertical shift of the sigmoid curve

    # Apply the custom sigmoid transformation
    output = a / (1 + tf.exp(-b * (x - c))) + d

    # Calculate the minimum and maximum values in the output tensor
    # These values are used for scaling the output
    min_output = tf.reduce_min(output)
    max_output = tf.reduce_max(output)

    # Scale the output to a range between 0 and 1
    # This is done by subtracting the min and dividing by the range (max - min)
    bounded_output = (output - min_output) / (max_output - min_output)

    return bounded_output

# polynomial fit
# Predefined coefficients for the polynomial
coeffs = [
    1.67128104e-01,  5.46585981e-01,  1.68540528e-01, -2.81036308e-01,
   -4.31316616e-02,  9.32131620e-02,  3.82093428e-03, -1.68218635e-02,
    2.75435107e-04,  1.77546605e-03, -8.66354619e-05, -1.14749692e-04,
    7.89867780e-06,  4.59828314e-06, -3.77395636e-07, -1.11373192e-07,
    1.01654033e-08,  1.49273718e-09, -1.46172688e-10, -8.49657331e-12,
    8.74134626e-13
]

# Define a closure that creates a custom polynomial activation function
def create_custom_polynomial_activation(coeffs):
    def custom_polynomial_activation(x):
        # Calculate polynomial output
        poly_output = sum(c * x**i for i, c in enumerate(coeffs))
        # Calculate min and max output for scaling
        min_output = tf.reduce_min(poly_output)
        max_output = tf.reduce_max(poly_output)
        # Scale the output to [0, 1]
        return (poly_output - min_output) / (max_output - min_output)
    return custom_polynomial_activation

# Use the closure to create the activation function
polynomial_activation = create_custom_polynomial_activation(coeffs)


# -- Neural Network Optimization Configuration --

optimizer = Adam(learning_rate=0.01)
opt = SGD(learning_rate=0.01)

# -- Neural Network Architecture --

# Define the neural network model
# The model will have two outputs: one for region and one for angle
# Input layer: Shape of the input data is set dynamically based on the number of regions
inputs = Input(shape=(num_regions,))
# Hidden layers: Two dense layers with number of neurons (units) each, using an activation function
x = Dense(12, activation=polynomial_activation)(inputs)
x = Dense(12, activation=polynomial_activation)(x)
# Output layer for region prediction: Number of units set dynamically based on number of region classes
region_output = Dense(num_region_classes, activation=custom_sigmoid, name='region_output')(x)
# Concatenation layer: Combines the output of the last hidden layer and the region_output
y = Concatenate()([x, region_output])
# Output layer for angle prediction: Number of units set dynamically based on number of angle classes
angle_output = Dense(num_angle_classes, activation=custom_sigmoid, name='angle_output')(y)
# Creation of the model: Defines the input and outputs for the model
model = Model(inputs=inputs, outputs=[region_output, angle_output])

# -- Model Compilation, Training, and Evaluation --

# Compile the model
# This step configures the model for training by setting the optimizer, loss function, and metrics for evaluation
model.compile(
    optimizer=opt,  # Use the SGD optimizer as defined earlier
    loss={
        'region_output': 'categorical_crossentropy',  # Loss function for the region output
        'angle_output': 'categorical_crossentropy'   # Loss function for the angle output
    },
    metrics={
        'region_output': 'accuracy',  # Evaluation metric for the region output
        'angle_output': 'accuracy'    # Evaluation metric for the angle output
    }
)

# Train the model
history = model.fit(
    features_train, 
    {'region_output': labels_regions_train, 'angle_output': labels_angles_train},
    validation_data=(features_val, {'region_output': labels_regions_val, 'angle_output': labels_angles_val}),
    epochs=100,  # Total number of iterations on the data
    batch_size=32  # Number of samples per gradient update
)

# Evaluate the model with original trained weights
results = model.evaluate(features_test, {'region_output': labels_regions_test, 'angle_output': labels_angles_test}, verbose=0)
print(f"Original Weights - Region Test Accuracy: {results[3]*100:.2f}%")
print(f"Original Weights - Angle Test Accuracy: {results[4]*100:.2f}%")

# -- Prepare for Stochastic Weight Inference --
# Extract trained weights and biases from the model
trained_weights_and_biases = [layer.get_weights() for layer in model.layers if len(layer.get_weights()) > 0]

# Function to sample new weights based on normal distribution around the trained weights
def sample_weights(trained_weights, std_factor=0.01):
    np.random.seed()  # Reset the NumPy random seed to ensure different results each run
    sampled_weights = []
    for weights in trained_weights:
        mean = weights[0]  # weights
        stddev = std_factor * np.abs(mean)  # Use absolute value to ensure non-negative standard deviation
        sampled_w = np.random.normal(loc=mean, scale=stddev, size=mean.shape)

        # If there is a bias term, sample for it as well
        if len(weights) > 1:
            mean_bias = weights[1]  # biases
            stddev_bias = std_factor * np.abs(mean_bias)
            sampled_b = np.random.normal(loc=mean_bias, scale=stddev_bias, size=mean_bias.shape)
            sampled_weights.append([sampled_w, sampled_b])
        else:
            sampled_weights.append([sampled_w])
    return sampled_weights

# Sample new weights for inference
sampled_weights = sample_weights(trained_weights_and_biases)

# Set sampled weights in the model for inference
for layer, new_weights in zip([l for l in model.layers if len(l.get_weights()) > 0], sampled_weights):
    layer.set_weights(new_weights)

# Evaluate the model with stochastic sampled weights
sampled_results = model.evaluate(features_test, {'region_output': labels_regions_test, 'angle_output': labels_angles_test}, verbose=0)
print(f"Sampled Weights - Region Test Accuracy: {sampled_results[3]*100:.2f}%")
print(f"Sampled Weights - Angle Test Accuracy: {sampled_results[4]*100:.2f}%")

# Plot Angle Accuracy
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman'] + matplotlib.rcParams['font.serif']
plt.figure(figsize=(10, 6))
plt.plot(history.history['angle_output_accuracy'], linewidth=2)
plt.plot(history.history['val_angle_output_accuracy'], linewidth=2)
plt.ylabel('Accuracy', fontsize=25)
plt.xlabel('Epoch', fontsize=25)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(['Train', 'Validation'], loc='upper left', fontsize=20)
plt.tight_layout()
plt.savefig('angle_accuracy.pdf', format='pdf', dpi=300)
plt.show()

# Plot Region Accuracy
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman'] + matplotlib.rcParams['font.serif']
plt.figure(figsize=(10, 6))
plt.plot(history.history['region_output_accuracy'], linewidth=2)
plt.plot(history.history['val_region_output_accuracy'], linewidth=2)
plt.ylabel('Accuracy', fontsize=25)
plt.xlabel('Epoch', fontsize=25)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(['Train', 'Validation'], loc='upper left', fontsize=20)
plt.tight_layout()
plt.savefig('region_accuracy.pdf', format='pdf', dpi=300)
plt.show()


# Calculate the normalized angle error (NAE) in the test set
# 1. Use the model to predict the regions and angles for the test set
predicted_values_test = model.predict(features_test)
predicted_regions_test = predicted_values_test[0]
predicted_angles_test = predicted_values_test[1]
predicted_region_labels_test = np.argmax(predicted_regions_test, axis=1)
predicted_angle_labels_test = np.argmax(predicted_angles_test, axis=1)
# 2. Import the true angle and the true region for the test set
true_angle_labels_test = np.argmax(labels_angles_test, axis=1)
true_region_labels_test = np.argmax(labels_regions_test, axis=1)
# 3. Convert relative angle labels to absolute angles
true_absolute_angles = true_region_labels_test*num_angle_classes + true_angle_labels_test*resolution
predicted_absolute_angles = predicted_region_labels_test*num_angle_classes + predicted_angle_labels_test*resolution
# 4. For each predicted absolute angle and the true absolute angle, calculate the angle error
angle_errors = []
for predicted, true in zip (predicted_absolute_angles, true_absolute_angles):
    error = abs(predicted - true)
    error = min(error, 360-error) # consider te circle nature for the angle of the source
    angle_errors.append(error)
# 5. Calculate the mean of these angle errors
mean_angle_error = np.mean(angle_errors) / resolution
angle_errors = [error / resolution for error in angle_errors]
#print(f"Normalized Angle Errors Vector: {angle_errors}")
print(f"Normalized Mean Angle Error : {mean_angle_error}")

