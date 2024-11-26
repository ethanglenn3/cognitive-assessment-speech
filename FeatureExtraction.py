# This code loads in the TAUKADIAL-24-train audio files from DementiaBank
# and extracts features that will be fed into a neural network for use
# in Cognitive Assessment. Users will be classified as No Cognitive Impairment
# or Mild Cognitive Impairment.

import os
import scipy
from scipy.io import wavfile
import scipy.io
import librosa as lb
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim

#-- Load Data -----------------------------------
# Loop through all `.wav` files in the directory
def loadWav(data_dir, tArray):
    for filename in os.listdir(data_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(data_dir, filename)
            
            # Load audio file using librosa
            audio, sr = lb.load(file_path, sr=None)  # sr=None preserves original sampling rate
            
            # Append audio data and sampling rate
            tArray.append((audio, sr))
    return tArray

# Initialize an empty list to store features
train = []
testTAUK = []
testAM = []

# Paths for datasets
train_dir = r'C:\Users\Ethan\OneDrive\Documents\College\EELE 578\Dataset\TAUKADIAL-24-Train\train'
testTAUK_dir = r'C:\Users\Ethan\OneDrive\Documents\College\EELE 578\Dataset\TAUKADIAL-24-Test\test'
testAM_dir = r'C:\Users\Ethan\OneDrive\Documents\College\EELE 578\Dataset\ADReSS-M-test-gr\test-gr'

# Convert to a NumPy array further processing
train = loadWav(train_dir,train)
testTAUK = loadWav(testTAUK_dir,testTAUK)
testAM = loadWav(testAM_dir,testAM)

train = np.array(train, dtype=object)
testTAUK = np.array(testTAUK, dtype=object)
testAM = np.array(testAM, dtype=object)

# Get Truth Labels
col = ['dx']
trainTruth = pd.read_csv(r'C:\Users\Ethan\OneDrive\Documents\College\EELE 578\Dataset\TAUKADIAL-24-Train\train\groundtruth.csv',usecols=col)
testTAUKTruth = pd.read_csv(r'C:\Users\Ethan\OneDrive\Documents\College\EELE 578\Dataset\TAUKADIAL-24-Test\test\TAUK-test-truth.csv',sep=';',usecols=col)
testAMTruth = pd.read_csv(r'C:\Users\Ethan\OneDrive\Documents\College\EELE 578\Dataset\ADReSS-M-test-gr\test-gr\test-gr-groundtruth.csv',usecols=col)

# Convert to NC to 0 and MCI to 1
trainTruth[trainTruth == 'NC'] = 0
trainTruth[trainTruth == 'MCI'] = 1
testTAUKTruth[testTAUKTruth == 'NC'] = 0
testTAUKTruth[testTAUKTruth == 'MCI'] = 1
testAMTruth[testAMTruth == 'Control'] = 0
testAMTruth[testAMTruth == 'ProbableAD'] = 1
#----------------------------------------------------

#-- Extract Features --------------------------------
# Split into Narrowband Frames
