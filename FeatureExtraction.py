# This code loads in the TAUKADIAL-24-train audio files from DementiaBank
# and extracts features that will be fed into a neural network for use
# in Cognitive Assessment. Users will be classified as No Cognitive Impairment
# or Mild Cognitive Impairment.

import os
import librosa as lb
import pandas as pd
import numpy as np
import parselmouth
from parselmouth.praat import call

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

# Function for framing an audio signal. frameSize corresponds to ms, overlap corresponds to %
def frame(audioRecording,samplingRate,frameSize,overlap):
    frameSamples = int(frameSize * samplingRate)
    stepSize = int(frameSamples * (1-overlap))
    numFrames = int((len(audioRecording) - frameSamples) / stepSize) + 1
    audioFrames = np.zeros((numFrames, frameSamples))

    hanningWindow = np.hanning(frameSamples)

    for i in range(numFrames):
        start = i*stepSize
        end = start + frameSamples

        # Ensure last frame is padded with zeros if it's shorter than others
        frameData = audioRecording[start:end]
        if len(frameData < frameSamples):
            frameData = np.pad(frameData, (0,frameSamples - len(frameData)),'constant')

        audioFrames[i] = frameData * hanningWindow
    
    return audioFrames

def pad_features(frames, max_frames):
    # Find the number of frames in the current recording
    num_frames = len(frames)
    
    # If the number of frames is less than max_frames, pad the frames with zeroed lists
    if num_frames < max_frames:
        # Calculate how many frames need to be added
        num_frames_to_add = max_frames - num_frames
        
        # Create the padding frames, each frame is a list of zeros of the same length as the first frame
        padding_frame = [0] * len(frames[0])  # A list of zeros with the same length as the first frame
        
        # Convert padding_frame to a 2D array (num_frames_to_add x len(frames[0]))
        padding_frame = np.array([padding_frame] * num_frames_to_add)

         # Append the padding frames to the frames list
        frames = np.concatenate([frames, padding_frame], axis=0)  # Concatenate along the first axis (rows)
    
    return frames

# Function for calculating energy
def extractEnergy(audioFrame):
    energy = np.sum(audioFrame**2) / len(audioFrame)
    return energy

# Function to calculate spectral flux
def calculateSpectralFlux(currentFrame, previousFrame):
    # Compute magnitude spectrum of both frames
    currentSpectrum = np.abs(np.fft.fft(currentFrame))
    previousSpectrum = np.abs(np.fft.fft(previousFrame))

    # Calculate the sum of the spectrums (for checking small values)
    currentSpectrumSum = np.sum(currentSpectrum)
    previousSpectrumSum = np.sum(previousSpectrum)

    if currentSpectrumSum < 1e-10 or previousSpectrumSum < 1e-10:
        return 0  # Return 0 if any spectrum sum is too small (near zero)

    # Calculate spectral flux
    flux = np.sum((currentSpectrum - previousSpectrum)**2)
    return flux

#-- Load Data -----------------------------------
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

# Normalize train
for i in range(len(train)):
    signal, rate = train[i]  # Extract signal and sampling rate
    max_val = np.max(np.abs(signal))
    if max_val != 0:  # Avoid division by zero
        train[i] = (signal / max_val, rate)  # Normalize signal and reassign

# Normalize testTAUK
for i in range(len(testTAUK)):
    signal, rate = testTAUK[i]  # Extract signal and sampling rate
    max_val = np.max(np.abs(signal))
    if max_val != 0:
        testTAUK[i] = (signal / max_val, rate)  # Normalize signal and reassign

# Normalize testAM
for i in range(len(testAM)):
    signal, rate = testAM[i]  # Extract signal and sampling rate
    max_val = np.max(np.abs(signal))
    if max_val != 0:
        testAM[i] = (signal / max_val, rate)  # Normalize signal and reassign

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
frameSize = 0.3     # 30 ms frames for narrowband
overlap = 0.5       # 50% overlap
trainFrames = []
testTAUKFrames = []
testAMFrames = []

for j in range(train.shape[0]):
    # Grab current audio recording sampling rate form array
    curTrain = train[j,0]
    curTrainSR = train[j,1]

    curFrame = frame(curTrain,curTrainSR,frameSize,overlap)

    trainFrames.append(curFrame)

for j in range(testTAUK.shape[0]):
    # Grab current audio recording sampling rate form array
    curTrain = testTAUK[j,0]
    curTrainSR = testTAUK[j,1]

    curFrame = frame(curTrain,curTrainSR,frameSize,overlap)

    testTAUKFrames.append(curFrame)

for j in range(testAM.shape[0]):
    # Grab current audio recording sampling rate form array
    curTrain = testAM[j,0]
    curTrainSR = testAM[j,1]

    curFrame = frame(curTrain,curTrainSR,frameSize,overlap)

    testAMFrames.append(curFrame)

# Calculate the maximum number of frames across all datasets
maxFrames = max(max(len(frames) for frames in trainFrames),
            max(len(frames) for frames in testTAUKFrames),
            max(len(frames) for frames in testAMFrames)
)

# Pad the dataset
for i in range(len(trainFrames)):
    trainFrames[i] = pad_features(trainFrames[i], maxFrames)

for i in range(len(testTAUKFrames)):
    testTAUKFrames[i] = pad_features(testTAUKFrames[i], maxFrames)

for i in range(len(testAMFrames)):
    testAMFrames[i] = pad_features(testAMFrames[i], maxFrames)

# 13 MFCCs, Corresponding Delta MFCCs
trainMFCCs, trainDeltaMFCCs = [], []
testTAUKMFCCs, testTAUKDeltaMFCCs = [], []
testAMMFCCs, testAMDeltaMFCCs = [], []

# Librosa's MFCC function automatically frames the recording and uses a hanning filter
# center must be set to false to ensure the same number of frames as the Frame function
for j in range(train.shape[0]):
    frameSamples = int(frameSize * train[j,1])
    stepSize = int(frameSamples * (1-overlap))
    mfccs = lb.feature.mfcc(y=train[j,0],sr=train[j,1],n_mfcc=13,n_fft=frameSamples,hop_length=stepSize,center=False)
    deltaMFCCS = lb.feature.delta(mfccs)
    trainMFCCs.append(mfccs)
    trainDeltaMFCCs.append(deltaMFCCS)

for j in range(testTAUK.shape[0]):
    frameSamples = int(frameSize * testTAUK[j,1])
    stepSize = int(frameSamples * (1-overlap))
    mfccs = lb.feature.mfcc(y=testTAUK[j,0],sr=testTAUK[j,1],n_mfcc=13,n_fft=frameSamples,hop_length=stepSize,center=False)
    deltaMFCCS = lb.feature.delta(mfccs)
    testTAUKMFCCs.append(mfccs)
    testTAUKDeltaMFCCs.append(deltaMFCCS)
    
for j in range(testAM.shape[0]):
    frameSamples = int(frameSize * testAM[j,1])
    stepSize = int(frameSamples * (1-overlap))
    mfccs = lb.feature.mfcc(y=testAM[j,0],sr=testAM[j,1],n_mfcc=13,n_fft=frameSamples,hop_length=stepSize,center=False)
    deltaMFCCS = lb.feature.delta(mfccs)
    testAMMFCCs.append(mfccs)
    testAMDeltaMFCCs.append(deltaMFCCS)

# Swap the MFCCs and Frames for fitting into Tensor
trainMFCCs = [np.swapaxes(mfccs, 0, 1) for mfccs in trainMFCCs]
trainDeltaMFCCs = [np.swapaxes(delta, 0, 1) for delta in trainDeltaMFCCs]

testTAUKMFCCs = [np.swapaxes(mfccs, 0, 1) for mfccs in testTAUKMFCCs]
testTAUKDeltaMFCCs = [np.swapaxes(delta, 0, 1) for delta in testTAUKDeltaMFCCs]

testAMMFCCs = [np.swapaxes(mfccs, 0, 1) for mfccs in testAMMFCCs]
testAMDeltaMFCCs = [np.swapaxes(delta, 0, 1) for delta in testAMDeltaMFCCs]

# Loop through and pad all MFCCs and deltaMFCCs
for i in range(len(trainMFCCs)):
    trainMFCCs[i] = pad_features(trainMFCCs[i], maxFrames)
    trainDeltaMFCCs[i] = pad_features(trainDeltaMFCCs[i], maxFrames)

for i in range(len(testTAUKMFCCs)):
    testTAUKMFCCs[i] = pad_features(testTAUKMFCCs[i], maxFrames)
    testTAUKDeltaMFCCs[i] = pad_features(testTAUKDeltaMFCCs[i], maxFrames)

for i in range(len(testAMMFCCs)):
    testAMMFCCs[i] = pad_features(testAMMFCCs[i], maxFrames)
    testAMDeltaMFCCs[i] = pad_features(testAMDeltaMFCCs[i], maxFrames)

# Energy & Spectral Flux
trainEnergy, trainFlux = [], []
testTAUKEnergy, testTAUKFlux = [], []
testAMEnergy, testAMFlux = [], []

for i in range(len(trainFrames)):
    curAudioEnergy, curAudioFlux = [], []
    curAudio = trainFrames[i]

    # Initialize the previous frame (all zeros for first frame)
    previousFrame = np.zeros(len(curAudio[0]))

    for j in range(len(curAudio)):
        # Energy
        frameEnergy = extractEnergy(curAudio[j])
        curAudioEnergy.append(frameEnergy)

        # Spectral Flux
        if j == 0 or j == 1:      
            flux = 0        # Remove NaN value of the first 2 frames
        else:
            flux = calculateSpectralFlux(curAudio[j], previousFrame)
            
        curAudioFlux.append(flux)

        # Update previous frame
        previousFrame = curAudio[j]

    trainEnergy.append(curAudioEnergy)
    trainFlux.append(curAudioFlux)

for i in range(len(testTAUKFrames)):
    curAudioEnergy, curAudioFlux = [], []
    curAudio = testTAUKFrames[i]

    # Initialize the previous frame (all zeros for first frame)
    previousFrame = np.zeros(len(curAudio[0]))

    for j in range(len(curAudio)):
        # Energy
        frameEnergy = extractEnergy(curAudio[j])
        curAudioEnergy.append(frameEnergy)

        # Spectral Flux
        if j == 0 or j == 1:         
            flux = 0        # Remove NaN value of the first frame
        else:
            flux = calculateSpectralFlux(curAudio[j], previousFrame)

        curAudioFlux.append(flux)

        # Update previous frame
        previousFrame = curAudio[j]

    testTAUKEnergy.append(curAudioEnergy)
    testTAUKFlux.append(curAudioFlux)

for i in range(len(testAMFrames)):
    curAudioEnergy, curAudioFlux = [], []
    curAudio = testAMFrames[i]

    # Initialize the previous frame (all zeros for first frame)
    previousFrame = np.zeros(len(curAudio[0]))

    for j in range(len(curAudio)):
        # Energy
        frameEnergy = extractEnergy(curAudio[j])
        curAudioEnergy.append(frameEnergy)

        # Spectral Flux
        if j == 0 or j == 1:         
            flux = 0        # Remove NaN value of the first frame
        else:
            flux = calculateSpectralFlux(curAudio[j], previousFrame)
            
        curAudioFlux.append(flux)

        # Update previous frame
        previousFrame = curAudio[j]

    testAMEnergy.append(curAudioEnergy)
    testAMFlux.append(curAudioFlux)

#-- Combine Features --------------------------------------------
# Ensure all feature arrays are NumPy arrays (if they aren't already)
trainMFCCs = np.array(trainMFCCs)
trainDeltaMFCCs = np.array(trainDeltaMFCCs)
trainEnergy = np.expand_dims(np.array(trainEnergy), axis=-1)
trainFlux = np.expand_dims(np.array(trainFlux), axis=-1)

testTAUKMFCCs = np.array(testTAUKMFCCs)
testTAUKDeltaMFCCs = np.array(testTAUKDeltaMFCCs)
testTAUKEnergy = np.expand_dims(np.array(testTAUKEnergy), axis=-1)
testTAUKFlux = np.expand_dims(np.array(testTAUKFlux), axis=-1)

testAMMFCCs = np.array(testAMMFCCs)
testAMDeltaMFCCs = np.array(testAMDeltaMFCCs)
testAMEnergy = np.expand_dims(np.array(testAMEnergy), axis=-1)
testAMFlux = np.expand_dims(np.array(testAMFlux), axis=-1)

# Feature tensors
trainFeatureTensor = np.concatenate([trainMFCCs, trainDeltaMFCCs, trainEnergy, trainFlux], axis=-1)
testTAUKFeatureTensor = np.concatenate([testTAUKMFCCs, testTAUKDeltaMFCCs, testTAUKEnergy, testTAUKFlux], axis=-1)
testAMFeatureTensor = np.concatenate([testAMMFCCs, testAMDeltaMFCCs, testAMEnergy, testAMFlux], axis=-1)

# Save the feature tensors as .npy files
np.save('trainFeatureTensor.npy', trainFeatureTensor)
np.save('testTAUKFeatureTensor.npy', testTAUKFeatureTensor)
np.save('testAMFeatureTensor.npy', testAMFeatureTensor)