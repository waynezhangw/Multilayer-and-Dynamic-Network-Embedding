import os
import time
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from numpy import zeros, float32 as REAL

from tqdm import tqdm_notebook
from sklearn.preprocessing import MinMaxScaler

# load_data function from MNE
def load_model(base_file, index2word_file):
    base = np.load(base_file)
    with open(index2word_file, 'r') as f:
        index2word = json.load(f)
    num_of_stops = 10626
    line_dim = 200
    vec_line = zeros((num_of_stops, line_dim), dtype=REAL)
    for i, value in enumerate(base):
        vec_line[int(index2word[i])-1] = value
    return vec_line  # 10626*200

# 2.load and pre-process the data
def load_rnn_data(data_dir):
    label_scalers = {}
    # read lable data
    df = pd.read_csv(data_dir + "20170405_countTransfer20-23.csv", header = None)
    label_sc = MinMaxScaler()
    labels = label_sc.fit_transform(df.values)
    labels = labels.reshape(-1, 1)
    label_sc.fit(df.iloc[:, 0].values.reshape(-1, 1))
    label_scalers = label_sc


    # read sequence data
    look_back = 5
    inputs = np.zeros((len(labels), look_back, 200))  # 200 is dim of the input features
    for i in range(0, 5):
        base = load_model(data_dir + 'base_' + str(i+1) + '.npy', data_dir + 'index2word_' + str(i+1) + '.json')
        sc = MinMaxScaler()
        data = sc.fit_transform(base)


        inputs[:, i, :] = data
        # inputs = inputs.reshape(-1, look_back, data.shape[1])

    test_portion_subway = int(0.1 * 199)+1
    test_portion_bus = int(0.1 * (len(inputs)-199))
    train_x = np.concatenate((inputs[test_portion_subway:199], inputs[199+test_portion_bus:]))
    train_y = np.concatenate((labels[test_portion_subway:199], labels[199+test_portion_bus:]))
    test_x = np.concatenate((inputs[:test_portion_subway], inputs[199:199+test_portion_bus]))
    test_y = np.concatenate((labels[:test_portion_subway], labels[199:199+test_portion_bus]))
    # train_x = inputs[test_portion_subway:199]
    # train_y = labels[test_portion_subway:199]
    # test_x = inputs[:test_portion_subway]
    # test_y = labels[:test_portion_subway]

    print("data pre-processing over:")
    print(train_x.shape)
    return train_x, train_y, test_x, test_y, label_scalers

def train(train_loader, learn_rate, hidden_dim=256, EPOCHS=1000000, model_type="GRU"):
    # Setting common hyperparameters
    input_dim = next(iter(train_loader))[0].shape[2]
    output_dim = 1
    n_layers = 2
    # Instantiating the models
    if model_type == "GRU":
        model = GRUNet(input_dim, hidden_dim, output_dim, n_layers)
    else:
        model = LSTMNet(input_dim, hidden_dim, output_dim, n_layers)
    model.to(device)

    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.train()
    print("Starting Training of {} model".format(model_type))
    epoch_times = []
    temp_epoch = 0
    # Start training loop
    for epoch in range(1, EPOCHS + 1):
        temp_epoch += 1
        start_time = time.perf_counter()
        h = model.init_hidden(batch_size)
        avg_loss = 0.
        counter = 0
        for x, label in train_loader:
            counter += 1
            if model_type == "GRU":
                h = h.data
            else:
                h = tuple([e.data for e in h])
            model.zero_grad()

            out, h = model(x.to(device).float(), h)
            loss = criterion(out, label.to(device).float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            # if counter % 20 == 0:
            #     print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter,
            #                                                                                len(train_loader),
            #                                                                                avg_loss / counter))
        current_time = time.perf_counter()
        if temp_epoch % 20 == 0:
            print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss / len(train_loader)))
            # print("Time Elapsed for Epoch: {} seconds".format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)
        if ((avg_loss / len(train_loader)) < 3.0e-5):
            print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss / len(train_loader)))
            break
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    return model

def evaluate(model, test_x, test_y, label_scalers):
    model.eval()
    outputs = []
    targets = []
    start_time = time.perf_counter()

    inp = torch.from_numpy(np.array(test_x))
    labs = torch.from_numpy(np.array(test_y))
    h = model.init_hidden(inp.shape[0])
    out, h = model(inp.to(device).float(), h)
    outputs.append(label_scalers.inverse_transform(out.cpu().detach().numpy()).reshape(-1))
    targets.append(label_scalers.inverse_transform(labs.numpy()).reshape(-1))

    print("Evaluation Time: {}".format(str(time.perf_counter()-start_time)))
    sMAPE = 0
    for i in range(len(outputs)):
        sMAPE += np.mean(abs(outputs[i]-targets[i])/(targets[i]+outputs[i])/2)/len(outputs)
    print("sMAPE: {}%".format(sMAPE*100))
    return outputs, targets, sMAPE

# 5.Next, we'll be defining the structure of the GRU and LSTM models. Both models have the same structure,
#  with the only difference being the recurrent layer (GRU/LSTM) and the initializing of the hidden state.
#  The hidden state for the LSTM is a tuple containing both the cell state and the hidden state, whereas the GRU only has a single hidden state.
class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden

# 1.Define the root directory of station vector data and label data
data_dir = "./data_GRU/"
files = os.listdir(data_dir)
print(files)
train_x, train_y, test_x, test_y, label_scalers = load_rnn_data(data_dir)

# 3.To improve the speed of our training, we can process the data in batches so that the model does not need to update its weights as frequently.
#   The Torch Dataset and DataLoader classes are useful for splitting our data into batches and shuffling them.
batch_size = 128
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

# 4.torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()
# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

lr = 0.001
gru_model = train(train_loader, lr, model_type="GRU")
lstm_model = train(train_loader, lr, model_type="LSTM")

# device = torch.device("cpu")
# gru_model.to(device)
# lstm_model.to(device)

gru_outputs, targets, gru_sMAPE = evaluate(gru_model, test_x, test_y, label_scalers)
lstm_outputs, targets, lstm_sMAPE = evaluate(lstm_model, test_x, test_y, label_scalers)

plt.figure(figsize=(14,10))
plt.subplot(2,2,1)
plt.plot(gru_outputs[0][-100:], "-o", color="g", label="Predicted")
plt.plot(targets[0][-100:], color="b", label="Actual")
plt.ylabel('Station transfer times')
plt.legend()

plt.subplot(2,2,2)
plt.plot(gru_outputs[0][-50:], "-o", color="g", label="Predicted")
plt.plot(targets[0][-50:], color="b", label="Actual")
plt.ylabel('Station transfer times')
plt.legend()

plt.subplot(2,2,3)
plt.plot(gru_outputs[0][:50], "-o", color="g", label="Predicted")
plt.plot(targets[0][:50], color="b", label="Actual")
plt.ylabel('Station transfer times')
plt.legend()

plt.subplot(2,2,4)
plt.plot(lstm_outputs[0][:100], "-o", color="g", label="Predicted")
plt.plot(targets[0][:100], color="b", label="Actual")
plt.ylabel('Station transfer times')
plt.legend()
plt.show()