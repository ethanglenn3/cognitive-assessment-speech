import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

X_test_tauk = np.load(r'./testTAUKFeatureTensor.npy')  # Shape: (recordings, frames, features)
X_test_am = np.load(r'./testAMFeatureTensor.npy')  # Shape: (recordings, frames, features)

# Load corresponding labels (0 for NC, 1 for MCI) and convert to 
label_encoder = LabelEncoder()

testTAUKTruth = label_encoder.fit_transform(np.load(r'./testTAUKTruth.npy', allow_pickle=True))
testAMTruth = label_encoder.fit_transform(np.load(r'./testAMTruth.npy', allow_pickle=True))

# Count 0s and 1s in each array
testTAUK_counts = np.unique(testTAUKTruth, return_counts=True)
testAM_counts = np.unique(testAMTruth, return_counts=True)

# Load saved model
model = tf.keras.models.load_model(r'C:\Users\Ethan\OneDrive\Documents\College\EELE 578\Project\GRU_2Layer64\speech_classification_rnn_model.keras')

# Predict tauk and am
tauk_pred = model.predict(X_test_tauk)
am_pred = model.predict(X_test_am)

# Convert probabilities to binary class labels
tauk_pred = (tauk_pred > 0.5).astype(int)
am_pred = (am_pred > 0.5).astype(int)

# Evaluate metrics for test data
tauktest_f1 = f1_score(testTAUKTruth, tauk_pred)
tauktest_accuracy = accuracy_score(testTAUKTruth, tauk_pred)
tauktest_recall = recall_score(testTAUKTruth, tauk_pred)
tauktest_precision = precision_score(testTAUKTruth, tauk_pred)

amtest_f1 = f1_score(testAMTruth, am_pred)
amtest_accuracy = accuracy_score(testAMTruth, am_pred)
amtest_recall = recall_score(testAMTruth, am_pred)
amtest_precision = precision_score(testAMTruth, am_pred)

# Compute recall for each class and calculate UAR for testTAUK
tauk_recall_0 = recall_score(testTAUKTruth, tauk_pred, pos_label=0)
tauk_recall_1 = recall_score(testTAUKTruth, tauk_pred, pos_label=1)
tauk_uar = (tauk_recall_0 + tauk_recall_1) / 2

# Compute recall for each class and calculate UAR for testAM
am_recall_0 = recall_score(testAMTruth, am_pred, pos_label=0)
am_recall_1 = recall_score(testAMTruth, am_pred, pos_label=1)
am_uar = (am_recall_0 + am_recall_1) / 2

# Metrics for table
tauk_metrics = {
    "F1-Score": tauktest_f1,
    "Accuracy": tauktest_accuracy,
    "Recall": tauktest_recall,
    "Precision": tauktest_precision,
    "UAR": tauk_uar
}

am_metrics = {
    "F1-Score": amtest_f1,
    "Accuracy": amtest_accuracy,
    "Recall": amtest_recall,
    "Precision": amtest_precision,
    "UAR": am_uar
}

# Create a DataFrame for comparison
metrics_table = pd.DataFrame([tauk_metrics, am_metrics], index=["Test TAUK", "Test AM"])

# Save the DataFrame to an Excel file
metrics_table.to_excel('model_performance_metrics_with_UAR.xlsx', index=True)