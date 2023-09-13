import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load your preprocessed data (assuming you've already prepared the data)
# X should contain the features, and y should contain the labels
# If you haven't created labels yet, you can omit the y part for now
# Replace 'X' and 'y' with your actual data

# load the prepared data into array for training and post process
X_file_path = './X_data.npy'
y_file_path = './y_labels.npy'
X = np.load(X_file_path)
y = np.load(y_file_path)

# X = ...
# y = ...

# Define the number of classes
num_classes = 3  # Replace with the actual number of classes

# One-hot encode your labels
y = tf.keras.utils.to_categorical(y, num_classes=num_classes)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Normalize the features using MinMaxScaler (if not done already)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Define the neural network architecture
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 100
batch_size = 32

history = model.fit(X_train, y_train, 
                    validation_data=(X_val, y_val), 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    verbose=2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')

# Save the trained model to a file
model.save('my_trained_model.h5')
print("Model saved to 'my_trained_model.h5'.")



# Plot training history (loss and accuracy)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
