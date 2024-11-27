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
import parselmouth
from parselmouth.praat import call
import torch
from torch import nn
from torch import optim

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

# Function for calculating energy
def extractEnergy(audioFrame):
    energy = np.sum(audioFrame**2) / len(audioFrame)
    return energy

# Function to calculate spectral flux
def calculateSpectralFlux(currentFrame, previousFrame):
    # Compute magnitude spectrum of both frames
    currentSpectrum = np.abs(np.fft.fft(currentFrame))
    previousSpectrum = np.abs(np.fft.fft(previousFrame))

    # Normalize magnitude spectra
    currentSpectrum /= np.sum(currentSpectrum)
    previousSpectrum /= np.sum(previousSpectrum)

    # Calculate spectral flux
    flux = np.sum((currentSpectrum - previousSpectrum)**2)
    return flux

# Function to calculate shimmer and jitter using Pareselmouth
def calcShimmerJitter(audioFrame, samplingRate):
    sound = parselmouth.Sound(audioFrame, samplingRate)
    f0min = 20
    f0max = int(samplingRate/2)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)

    # Calculate Jitter (local) and Jitter (rap) using Praat's call function
    jitterRap = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)

    # Calculate Shimmer (local) and Shimmer (apq11) using Praat's call function
    shimmerApq11 = call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    return jitterRap, shimmerApq11

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
        if j == 1:      
            flux = 0        # Remove NaN value of the first frame
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
        if j == 1:      
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
        if j == 1:      
            flux = 0        # Remove NaN value of the first frame
        else:
            flux = calculateSpectralFlux(curAudio[j], previousFrame)
            
        curAudioFlux.append(flux)

        # Update previous frame
        previousFrame = curAudio[j]

    testAMEnergy.append(curAudioEnergy)
    testAMFlux.append(curAudioFlux)

# Shimmer & Jitter
trainJitterRap, trainShimmerApq = [], []
testTAUKJitterRap, testTAUKShimmerApq = [], []
testAMFJitterRap, testAMShimmerApq = [], []


for i in range(len(trainFrames)):
    curJitterRap = []
    curShimmerApq = []
    curAudio = trainFrames[i]
    samplingRate = train[i,1]

    for j in range(len(curAudio)):
        # Calculate jitter and shimmer for the current frame
        jitterRap, shimmerApq11 = calcShimmerJitter(curAudio[j], samplingRate)

        # Append the calculated values to the corresponding lists
        curJitterRap.append(jitterRap)
        curShimmerApq.append(shimmerApq11)

    # After processing all frames for this recording, append the results
    trainJitterRap.append(curJitterRap)
    trainShimmerApq.append(curShimmerApq)

for i in range(len(testTAUKFrames)):
    curJitterRap = []
    curShimmerApq = []
    curAudio = trainFrames[i]
    samplingRate = train[i,1]

    for j in range(len(curAudio)):
        # Calculate jitter and shimmer for the current frame
        jitterRap, shimmerApq11 = calcShimmerJitter(curAudio[j], samplingRate)

        # Append the calculated values to the corresponding lists
        curJitterRap.append(jitterRap)
        curShimmerApq.append(shimmerApq11)

    # Append the results for the current recording
    testTAUKJitterRap.append(curJitterRap)
    testTAUKShimmerApq.append(curShimmerApq)

for i in range(len(testAMFrames)):
    curJitterRap = []
    curShimmerApq = []
    curAudio = trainFrames[i]
    samplingRate = train[i,1]

    for j in range(len(curAudio)):
        # Calculate jitter and shimmer for the current frame
        jitterRap, shimmerApq11 = calcShimmerJitter(curAudio[j], samplingRate)

        # Append the calculated values to the corresponding lists
        curJitterRap.append(jitterRap)
        curShimmerApq.append(shimmerApq11)

    # Append the results for the current recording
    testAMFJitterRap.append(curJitterRap)
    testAMShimmerApq.append(curShimmerApq)