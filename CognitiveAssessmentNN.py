import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Masking
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# Load your feature tensors and labels
# Replace these file paths with your actual file paths
X_train = np.load(r'./trainFeatureTensor.npy')  # Shape: (recordings, frames, features)
X_test_tauk = np.load(r'./testTAUKFeatureTensor.npy')  # Shape: (recordings, frames, features)
X_test_am = np.load(r'./testAMFeatureTensor.npy')  # Shape: (recordings, frames, features)

# Load corresponding labels (0 for NC, 1 for MCI) and convert to 
label_encoder = LabelEncoder()

trainTruth = label_encoder.fit_transform(np.load(r'./trainTruth.npy', allow_pickle=True))
testTAUKTruth = label_encoder.fit_transform(np.load(r'./testTAUKTruth.npy', allow_pickle=True))
testAMTruth = label_encoder.fit_transform(np.load(r'./testAMTruth.npy', allow_pickle=True))

# Split 50 recordings from X_train for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, trainTruth, test_size=50, random_state=42)

# Define the neural network model
model = Sequential()

# Add a Masking layer to ignore padded values (assuming padding is 0)
model.add(Masking(mask_value=0.0, input_shape=(X_train_split.shape[1], X_train_split.shape[2])))

# Adding an RNN layer
model.add(GRU(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.2))  # Adding dropout for regularization
model.add(GRU(64, activation='tanh', return_sequences=False))
model.add(Dropout(0.2))  # Adding dropout for regularization


# Adding a Dense layer with a sigmoid activation for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model using the split validation set
model.fit(X_train_split, y_train_split, epochs=25, batch_size=32, validation_data=(X_val_split, y_val_split))

# Evaluate the model on the testTAUK set
test_tauk_pred = model.predict(X_test_tauk)
test_tauk_loss, test_tauk_acc = model.evaluate(X_test_tauk, testTAUKTruth)

# Confusion matrix
test_tauk_pred = (test_tauk_pred > 0.5).astype(int)
tauk_cm = confusion_matrix(testTAUKTruth, test_tauk_pred)
print(tauk_cm)
np.savetxt("tauk_cm.csv", tauk_cm, delimiter=",")

# Evaluate the model on the testAM set
test_am_pred = model.predict(X_test_am)
test_am_loss, test_am_acc = model.evaluate(X_test_am, testAMTruth)

# Confusion matrix
test_am_pred = (test_am_pred > 0.5).astype(int)
am_cm = confusion_matrix(testAMTruth, test_am_pred)
print(am_cm)
np.savetxt("am_cm.csv", am_cm, delimiter=",")

# Visualizing TestTAUK Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(tauk_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['NC', 'MCI'], yticklabels=['NC', 'MCI'])
plt.title('Confusion Matrix for TestTAUK')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Save the TestTAUK confusion matrix plot
plt.savefig('tauk_confusion_matrix.png')  # Saves as a PNG file
plt.close()  # Closes the figure to free up memory

# Visualizing TestAM Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(am_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['NC', 'MCI'], yticklabels=['NC', 'MCI'])
plt.title('Confusion Matrix for TestAM')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Save the TestAM confusion matrix plot
plt.savefig('am_confusion_matrix.png')  # Saves as a PNG file
plt.close()  # Closes the figure to free up memory

# Save the loss and accuracy results in a dictionary
results = {
    'TestTAUK Loss': test_tauk_loss,
    'TestTAUK Accuracy': test_tauk_acc,
    'TestAM Loss': test_am_loss,
    'TestAM Accuracy': test_am_acc
}

# Convert the dictionary to a pandas DataFrame
results_df = pd.DataFrame([results])

# Save the results to a CSV file
results_df.to_csv('model_evaluation_results.csv', index=False)

# Optionally, save the model for later use
model.save('speech_classification_rnn_model.keras')