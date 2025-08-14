import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import glob
import re
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from pytorch_ranger import Ranger
from sklearn.model_selection import KFold
import torchvision.models as models
from sklearn.model_selection import KFold
from tqdm import tqdm 
from pytorch_ranger import Ranger
from collections import Counter
import os

import pickle


from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch.nn.functional as F 
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt




def fill_frames(frame_num, file_id, df):
    # Handle NaN in 'frame' column
    max_frame = int(df['frame'].max()) if not pd.isna(df['frame'].max()) else 0

    # Create a set of existing frames
    existing_frames = set(df['frame'].dropna().astype(int))
    all_frames = set(range(max_frame + 1))
    missing_frames = sorted(all_frames - existing_frames)

    # Function to insert a new frame into the DataFrame
    def insert_frame(frame_value):
        new_row = pd.DataFrame([{
            'frame': frame_value,'DetObj#': 0,'x': 0,'y': 0,'z': 0,'v': 0,'source_file_id': file_id
        }])
        return new_row

    # Fill missing frames
    i = 0
    while i < len(missing_frames):
        frame_to_fill = missing_frames[i]
        if frame_to_fill == 0 or (frame_to_fill - 1 in existing_frames):
            index_tochange = df[df['frame'] == frame_to_fill - 1].index[-1] if frame_to_fill > 0 else -1
            new_df = insert_frame(frame_to_fill)
            df = pd.concat([df.iloc[:index_tochange + 1], new_df, df.iloc[index_tochange + 1:]]).reset_index(drop=True)
            existing_frames.add(frame_to_fill)  # Update existing frames
        i += 1

    # Fill frames up to frame_num
    current_frame = max_frame + 1
    while current_frame <= frame_num:
        new_df = insert_frame(current_frame)
        df = pd.concat([df, new_df], ignore_index=True).reset_index(drop=True)
        current_frame += 1

    return df




def insert_frame_rows(frame_value, file_id, start_index, num_missing):
    new_rows = pd.DataFrame({
        'frame': frame_value,
        'DetObj#': range(start_index + 1, start_index + num_missing + 1),'x': 0,'y': 0,'z': 0,'v': 0,
        'source_file_id': file_id
    })
    return new_rows

def fill_DetObj(maxpoints, file_id, df):
    for i in range(df['frame'].max() + 1):
        df_thisframe = df[df['frame'] == i]
        thisframe_maxpoints = df_thisframe['DetObj#'].max()

        if thisframe_maxpoints < maxpoints:
            index_tochange = df_thisframe.index[-1] if not df_thisframe.empty else -1
            num_missing = maxpoints - thisframe_maxpoints

            new_rows = insert_frame_rows(i, file_id, thisframe_maxpoints, num_missing)
            df = pd.concat([df.iloc[:index_tochange + 1], new_rows, df.iloc[index_tochange + 1:]]).reset_index(drop=True)

    return df


num_samples = 732       #732    537
sequence_length = 30
num_features = 4
num_detobj = 21     # 17 21
input_size = (num_detobj + 1) * 4
num_classes = 6
hidden_size = 128 #  64    128    256
num_layers = 3    




class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.2):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))  # LSTM forward pass
        out = self.dropout(out[:, -1, :])  # Take the last time step output
        out = self.fc(out)  # Fully connected layer
        return out



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = RNN_LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
model.load_state_dict(torch.load('/Users/zuhayrrahman/Documents/Semester B 24 25/CS4514 Project/Project/code/RNNModel_FinalReport.pth', weights_only=True))


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.eval()  # Set the model to evaluation mode


# Obtaining prediction
newfile = pd.read_csv('/Users/zuhayrrahman/Documents/Semester B 24 25/CS4514 Project/Project/My Data/Final (Motion & No Motion)/demos/script test dat files/newfile.csv')
newfile['source_file_id'] = 0
newfile = fill_frames(29, 0, newfile)
newfile = fill_DetObj(21, 0, newfile)
newfile = newfile.sort_values(by=['frame', 'DetObj#']).reset_index(drop=True)
input_arr = newfile[['x', 'y', 'z', 'v']].values
new = input_arr.reshape(1, 30, 22, 4)
input_matrix = new.reshape(1, 30, 22*4)

model.eval()

input_matrix = np.array(input_matrix, dtype=np.float32)  # Convert to float32
input_matrix = torch.tensor(input_matrix, dtype=torch.float32).to(device)

with torch.no_grad():
    pred_value = model(input_matrix)
    _, predicted = torch.max(pred_value, 1)

# Map predicted class to motion type
motion_classes = {
    0: 'No Motion',
    1: 'Bowing',
    2: 'Waving',
    3: 'Jumping',
    4: 'Moving Body',
    5: 'Walking'
}

predicted_motion = motion_classes[predicted.item()]

print(predicted_motion)
