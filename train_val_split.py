'''
Split the validation and training set of the data and the labels
'''

import numpy as np
import torch
from sklearn.model_selection import train_test_split

dataset = np.load('./data.npy', allow_pickle=True)
labels = np.load('./labels2.npy')

np.random.seed(100)
torch.manual_seed(100)
seed = 100

data_train, data_valid, label_train, label_valid = train_test_split(dataset, labels, test_size=0.1, random_state=seed)