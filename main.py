import argparse
from time import time

import numpy as np
import pandas as pd
import pyquist as pq
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import DataLoader

from model import MultiLayerPerceptron
from dataset import WindyDataset
from visualize_data import visualize_data
# from train_val_split import data_train, data_valid, label_train, label_valid


""" Wind Data Classification

In this project we will build our own neural network pipeline to do classification on the wind dataset. The ultimate 
goal is to be able to classify between "good" turbine data and "bad".

"""
np.random.seed(100)
torch.manual_seed(100)
seed = 100

# =================================== LOAD DATASET =========================================== #
column_names = pq.io.read_json('./column_names.json')

file_pattern = './data/*.xlsx'
filenames = pq.io.match(file_pattern)

raw_data = pd.DataFrame()
for name in filenames:
    print(name)
    data = pq.io.read_excel(name, sheet_name='Main', header=6, usecols=column_names, engine='openpyxl')
    raw_data = raw_data.append(data, ignore_index=True, sort=False,)
print('Done reading Files!')


# ===================================== DATA CLEANING ========================================== #

# breakpoint()
# Remove unwanted columns

# isna = raw_data.isna().any()
# empty_cols = list(raw_data.columns[isna].values)
# empty_cols.remove('TON')            # don't remove TON and Background partially empty columns
# empty_cols.remove('Background')
#
# unweighted_cols = raw_data.loc[:, raw_data.columns.str.endswith('-u')].columns.tolist()
#
# # empty_cols = list(set(empty_cols))      # get unique list
# # unweighted_cols = list(set(unweighted_cols))
#
# duplicate_columns = ['Project', 'TIMESTAMP', 'WSpd_Avg [m/s]', 'WDir_Avg [°]', 'Humidity [%]', 'Temperature [°C]',
#                      'rain before?', 'rain after?', 'Pressure (Pa)', 'Amb. Rain Gauge', 'Rain?', 'Wind Direction']
# non_useful_columns = ['Start Time', 'Turbine Power Verified?', 'Rain_Tot [mm]', 'Wind Dir. avg', 'Index',
#                       'Amb. Temperature', 'Amb. Humidity', 'Amb. Pressure', 'Comments', 'Rain?', 'Gust?', 'Low Temp?',
#                       'Time?', 'Leq-L90 > Threshold?','Max SPL Threshold?', 'Min SPL Threshold?', 'T Power > Threshold?',
#                       'Yaw Angle (°)', 'Turbines ON/OFF', 'Turbine Power Verified?']
#
# drop_cols = empty_cols + unweighted_cols + duplicate_columns + non_useful_columns
# drop_cols = list(set(drop_cols))        # get unique column names
#
# raw_data = raw_data.drop(columns=drop_cols, axis=1)

# ## Add column 'Junk' to distinguish between good and bad data
raw_data['Junk?'] = raw_data['Junk?'].str.lower().fillna('no')

# print(f"Through cleaning of data, we have eliminated {len(drop_cols)} columns.")

# ===================================== DATA VISUALIZATION ===================================== #

junk_data, useable_TON, useable_back = raw_data[raw_data['Junk?']=='yes'], raw_data[raw_data['Useable TON Data?']==1], \
                                       raw_data[raw_data['Useable Background Data?']==1]

if len(raw_data) != len(junk_data) + len(useable_TON) + len(useable_back):
    print(f'Not the same length! ({len(raw_data)} vs. {len(junk_data) + len(useable_TON) + len(useable_back)})')

# projects_junk = junk_data['Project Name'].value_counts()
# projects_TON = useable_TON['Project Name'].value_counts()
# projects_back = useable_back['Project Name'].value_counts()
# projects_all = pd.concat((projects_junk, projects_TON, projects_back), axis=1, sort=True)
# projects_all.columns = ['Projects from Junk Data', 'Projects from Useable TON Data', 'Projects from Useable Background Data']
#
# junk_data.drop(columns='Project Name', axis=1, inplace=True)
# useable_TON.drop(columns='Project Name', axis=1, inplace=True)
# useable_back.drop(columns='Project Name', axis=1, inplace=True)

# Save stats
#  pq.io.write_excel('./data_stats.xlsx', {'good':stats_good, 'bad':stats_bad})

# pie_plot = projects_all.plot.pie(subplots=True, figsize=(20,10), title='Breakdown of Projects Used in Datasets')
# pie_plot[0].get_figure().savefig('./pie_plot.png', bbox_inches="tight")

# =================================== BALANCE DATASET =========================================== #

# We want the same number of good data and bad data
print(f"Junk Data points: {len(junk_data)} \n"
      f"Good TON Data points: {len(useable_TON)} \n"
      f"Good Background Data points: {len(useable_back)} \n")

# datasets = [junk_data, useable_TON, useable_back]
# smallest_dataset_idx = np.argmin([len(junk_data), len(useable_TON), len(useable_back)])
# smallest_dataset = datasets[smallest_dataset_idx]

datasets = [junk_data, useable_TON]
smallest_dataset_idx = np.argmin([len(junk_data), len(useable_TON)])
smallest_dataset = datasets[smallest_dataset_idx]


data_all = pd.DataFrame()
for i, dataset in enumerate(datasets):
    new_dataset = dataset.sample(n=len(smallest_dataset), random_state=seed)
    new_dataset['label'] = i
    data_all = data_all.append(new_dataset)
data_all = data_all.sample(frac=1)


# =================================== DATA PREPROCESSING =========================================== #

# We will preprocess the continuous features by normalizing against the feature mean and standard deviation
# Let's also drop all categorical columns and use the junk column as the labels of the data

# NORMALIZE CONTINUOUS FEATURES
continuous_feats = data_all.select_dtypes(include=[np.number]).columns
cont_data = data_all[continuous_feats]
cont_norm = ((cont_data - cont_data.mean())/cont_data.std())

# CATEGORICAL FEATURES
label_column = data_all['label'].reset_index(drop=True)   # use as labels later
np.save('./labels.npy', label_column)

data_clean = cont_norm
data_clean = cont_norm.drop(columns=['Useable TON Data?', 'Useable Background Data?'])
data_clean = data_clean.drop(columns=['label'], )
data_clean = data_clean.drop(columns=['TON', 'Background'])
data_clean = data_clean.dropna(axis=1)
np.save('./data.npy', data_clean)

# breakpoint()

# data_clean = cont_norm.reset_index(drop=True)   # reset index bc currently the index is the wind farm name


# ================================ MAKE THE TRAIN AND VAL SPLIT ====================================== #
# we'll make use of the train_test_split method to randomly divide our dataset into two portions
# we can also control the relative sizes of the two splits using the test_size parameter

feat_train, feat_valid, label_train, label_valid = train_test_split(data_clean.values, label_column.values, random_state=seed)

# ==================================== LOAD DATA AND MODEL =========================================== #

def load_data(batch_size):
    train_dataset = WindyDataset(feat_train, label_train)
    valid_dataset = WindyDataset(feat_valid, label_valid)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def load_model(lr):
    loss_fnc = torch.nn.BCELoss()
    model = MultiLayerPerceptron(feat_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, loss_fnc, optimizer


def evaluate(model, val_loader):
    total_corr = 0
    count = 0
    label_encoder = LabelEncoder()

    for i, vbatch in enumerate(val_loader):
        count += 1
        feats, label = vbatch
        feats = feats.float()
        # labels = np.asarray(label)
        # labels = label_encoder.fit_transform(labels)
        # labels = torch.from_numpy(labels)

        predictions = model.forward(feats)
        # outputs = outputs.detach().numpy()
        # predictions = outputs.argmax(axis=1)
        corr = (predictions > 0.5).squeeze().long() == label
        total_corr += int(corr.sum())
    print(total_corr)
    return float(total_corr) / (len(val_loader.dataset))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--eval_every', type=int, default=10)

    args = parser.parse_args()

    MaxEpochs = args.epochs
    lr = args.lr
    batchsize = args.batch_size
    eval_every = args.eval_every

    model, loss_fnc, optimizer = load_model(lr=lr)
    train_loader, val_loader = load_data(batchsize)
    label_encoder = LabelEncoder()
    valid_acc_array = []
    train_array = []
    # breakpoint()
    t = 0
    t_array = []
    t_0 = time()
    time_array = []
    for epoch in range(MaxEpochs):
        accum_loss = 0
        tot_corr_sum = 0

        for i, batch in enumerate(train_loader):
            # this gets one "batch" of data
            feats, label = batch  # feats will have shape (batch_size,4)
            # labels = np.asarray(labels)
            # labels = label_encoder.fit_transform(labels)

            # need to send batch through model and do a gradient opt step;
            # first set all gradients to zero
            optimizer.zero_grad()

            # Run the neural network model on the batch, and get answers
            feats = feats.float()
            outputs = model(feats)  # has shape (batch_size,1)
            # labels = torch.from_numpy(labels)

            # compute the loss function (BCE as above) using the correct answer for the entire batch
            # label was an int, needs to become a float
            batch_loss = loss_fnc(input=outputs.squeeze(), target=label.float())
            accum_loss += batch_loss

            # computes the gradient of loss with respect to the parameters
            # to make this possible;  uses back-propagation
            batch_loss.backward()

            # Change the parameters  in the model with one 'step' guided by the learning rate.
            # Recall parameters are the weights & bias
            optimizer.step()

            # calculate number of correct predictions
            # outputs = outputs.detach().numpy()
            # predictions = outputs.argmax(axis=1)
            predictions=outputs
            corr = (predictions > 0.5).squeeze().long() == label
            tot_corr_sum += int(corr.sum())

        valid_acc = evaluate(model, val_loader)
        print("Epoch: {} | Total Correct: {}| Test acc: {}".format(epoch+1, tot_corr_sum, valid_acc))
        train_acc = tot_corr_sum/(len(train_loader.dataset))
        t_array.append(t)
        time_array.append(time()-t_0)

        train_array.append(train_acc)
        valid_acc_array.append(valid_acc)

        t = t+1

    print("training loop time: ", time()-t_0)
    print("bs: ", batchsize, "max val acc:", max(valid_acc_array))
    print('total data size:', len(train_loader.dataset))
    print(max(train_array))
    print(max(valid_acc_array))


    # Plot Validation and Training Data
    plt.figure(figsize=(20,10))
    plt.title(f"Validation and Training Accuracy Over Number of Gradient Steps (lr:{lr}, bs:{batchsize})")
    plt.plot(t_array, train_array, label="Training")
    plt.plot(t_array, valid_acc_array, label="Validation")
    plt.xlabel("Number of Gradient Steps")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.savefig(f"ValandTrain_lr_{lr}_bs_{batchsize}.png")

    # Plot with smooth training data
    # from scipy.signal import savgol_filter
    # plt.figure()
    # plt.title("Smoothed training data")
    # plt.plot(t_array, savgol_filter(train_array, 3, 1), label="Training")
    # plt.plot(t_array, valid_acc_array, label="Validation")
    # plt.xlabel("Number of Gradient Steps")
    # plt.ylabel("Accuracy")
    # plt.legend(loc='best')
    # plt.savefig("Smooth_ValandTrain_LAST.png")



if __name__ == "__main__":
    main()