import numpy as np
import pandas as pd

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

# Save DataFrames as numpy arrays
np.save(r'C:\Users\Ethan\OneDrive\Documents\College\EELE 578\Dataset\TAUKADIAL-24-Train\trainTruth.npy', trainTruth.to_numpy())
np.save(r'C:\Users\Ethan\OneDrive\Documents\College\EELE 578\Dataset\TAUKADIAL-24-Test\testTAUKTruth.npy', testTAUKTruth.to_numpy())
np.save(r'C:\Users\Ethan\OneDrive\Documents\College\EELE 578\Dataset\ADReSS-M-test-gr\testAMTruth.npy', testAMTruth.to_numpy())