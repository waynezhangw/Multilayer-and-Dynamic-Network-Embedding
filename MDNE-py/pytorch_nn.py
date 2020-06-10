# This python file is used as a predictor (a fully connected neural network)
# input(200)-hidden layer(100 50 20)-output(1)
# Author: Wenyuan Zhang
# Time: 2020/04/08

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
import torch.nn.functional as F

from tqdm import tqdm_notebook
from sklearn.preprocessing import MinMaxScaler

# load vector representation results from MNE
def load_MNE_model(base_file, index2word_file):
    base = np.load(base_file)
    with open(index2word_file, 'r') as f:
        index2word = json.load(f)
    num_of_stops = 10626
    line_dim = 200
    vec_line = zeros((num_of_stops, line_dim), dtype=REAL)
    for i, value in enumerate(base):
        vec_line[int(index2word[i])-1] = value
    return vec_line  # 10626*200

# load edge data
def load_lable():
    df = pd.read_csv(data_dir + "20170405_edgeType20-23.csv", header='infer')
    df = df.sort_values(by=['num'], ascending=False)
    # plt.plot(np.log(df.values[:-1,3]), "-", color="g", label="Predicted")
    threshold_transfer1 = 100
    threshold_transfer2 = 300
    who_to_predict = 3
    valid_transfer = [index for index, w in enumerate(df.values) if w[3] > threshold_transfer1 and w[3] < threshold_transfer2]
    head = df.values[valid_transfer,1]
    head = head.reshape(-1, 1)
    tail = df.values[valid_transfer,2]
    tail = tail.reshape(-1, 1)

    sc = MinMaxScaler()
    label_sc = MinMaxScaler()
    label_sc.fit(df.iloc[:, who_to_predict].values.reshape(-1, 1))
    data = sc.fit_transform(df.values)
    label_scalers = label_sc

    edge_labels = data[valid_transfer, who_to_predict]    # read transfer times and cost
    edge_labels = edge_labels.reshape(-1, 1)
    return head, tail, edge_labels, label_scalers

def load_lable2():
    df = pd.read_csv(data_dir + "20170405_countTransfer20-23.csv", header='infer')
    df['index'] = np.arange(0, 10626)
    df = df.sort_values(by=['num'], ascending=False)
    sc = MinMaxScaler()
    label_sc = MinMaxScaler()
    label_sc.fit(df.iloc[:, 0].values.reshape(-1, 1))
    data = sc.fit_transform(df.values)
    label_scalers2 = label_sc

    threshold_num_small = 2000
    threshold_num_big = 4000
    transfer_num = data[threshold_num_small:threshold_num_big, 0]
    transfer_num = transfer_num.reshape(-1, 1)
    transfer_accumTime = data[threshold_num_small:threshold_num_big, 1]
    transfer_accumTime = transfer_accumTime.reshape(-1, 1)
    train_x_index = df.values[threshold_num_small:threshold_num_big, 2]
    return transfer_num, transfer_accumTime, train_x_index.astype(int), label_scalers2

# 2.load and pre-process the data
def load_rnn_data(data_path):
    df = pd.read_csv(data_path, header = None)
    vec_rnn = df.values
    return vec_rnn

def train(train_loader, learn_rate, EPOCHS=2000):  #******************************************************************
    model = NeuralNet().to(device)
    # print(model)
    # Defining loss function and optimizer
    criterion = nn.MSELoss(reduction = 'sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=1e-5)

    model.train()
    epoch_times = []
    temp_epoch = 0
    # Start training loop
    for epoch in range(1, EPOCHS + 1):
        temp_epoch += 1
        start_time = time.perf_counter()
        avg_loss = 0
        counter = 0
        for x, label in train_loader:
            counter += 1
            # Move tensors to the configured device
            x = x.to(device).float()
            label = label.to(device).float()

            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, label)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

            # if counter % 20 == 0:
            #     print("Epoch {}......Step: {}/{}....... Loss for Step: {}".format(epoch, counter,
            #                                                                                len(train_loader),
            #                                                                                loss.item()))
        current_time = time.perf_counter()
        if epoch % 1000 == 0:
            print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss / len(train_loader)))
        epoch_times.append(current_time - start_time)
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    return model

def evaluate(model, test_x, test_y, label_scalers):
    model.eval()
    start_time = time.perf_counter()
    outputs = model(test_x.to(device).float())
    outputs = outputs.cpu().detach().numpy()
    test_y = test_y.cpu().detach().numpy()

    sMAPE = 0
    test0_counter = 0
    for i in range(len(outputs)):
        if (test_y[i] == 0):
            test0_counter += 1
            continue
        sMAPE += np.mean(abs(outputs[i]-test_y[i])/(abs(test_y[i]+outputs[i])/2))

    outputs_y = label_scalers.inverse_transform(outputs.reshape(-1,1))
    real_y = label_scalers.inverse_transform(test_y.reshape(-1,1))
    sMAPE = sMAPE/(len(outputs)-test0_counter)
    print("sMAPE: {}%".format(sMAPE*100))
    # print("Evaluation Time: {}".format(str(time.perf_counter() - start_time)))
    return outputs_y, real_y, sMAPE

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(200, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 20)
        self.fc4 = nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

def get_LinearRegData(station_rep, head, tail):
    head = (head-1).reshape(-1)
    tail = (tail-1).reshape(-1)
    big_head = station_rep[head.astype(int), :]
    big_tail = station_rep[tail.astype(int), :]
    # final_x = np.add(big_head, big_tail)/2
    # final_x = np.multiply(big_head, big_tail)
    final_x = abs(np.subtract(big_head, big_tail))
    # final_x = abs(np.subtract(big_head, big_tail))**2
    # final_x = np.sum(final_x, axis=1)
    return final_x

def divide_dataset(cut_ratio, x, y):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    batch_size = x.shape[0]
    train_data = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=False)
    for shuffled_x, shuffled_y in train_loader:
        test_portion = int(cut_ratio * batch_size)+1
        train_x = shuffled_x[:test_portion]
        train_y = shuffled_y[:test_portion]
        test_x = shuffled_x[test_portion:batch_size]
        test_y = shuffled_y[test_portion:batch_size]
        # use all data to train
        # train_x = shuffled_x
        # train_y = shuffled_y
        break
    return train_x, train_y, test_x, test_y

def find_ratio_of_topn(x1, n):
    my_original = np.copy(x1)
    x1.sort()
    top_n = 0
    for i in range(1, n+1):
        temp_v = x1[-i]
        temp_index = np.where(my_original == temp_v)
        if temp_index[0][0] < n:
            top_n += 1

    return top_n/n

def show_compare_curve(cut_ratio, MAPE):
    plt.figure(figsize=(11, 10))
    plt.plot(cut_ratio, MAPE[0,:], "-o", color="g", label="GRU1")
    plt.plot(cut_ratio, MAPE[1,:], "-o", color="b", label="GRU2")
    plt.plot(cut_ratio, MAPE[2,:], "-o", color="m", label="Single")
    plt.plot(cut_ratio, MAPE[3, :], "-o", color="y", label="Average")
    plt.legend(prop={'size': 16})
    plt.xlabel('fraction of labeled edges', fontsize=23)
    plt.ylabel('MAPE', fontsize=23)
    plt.tick_params(labelsize=20)
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # 1.Define the root directory of station vector data and label data
    data_dir = "./data_GRU/"
    head, tail, edge_labels, label_scalers = load_lable()
    base20_23 = load_MNE_model(data_dir + 'base.npy', data_dir + 'index2word.json')
    rnn_rep = load_rnn_data(data_dir + "merged_rep_gru-250epoch-200.csv")
    rnn_rep2 = load_rnn_data(data_dir + "merged_rep_gru-hihj-38epoch.csv")
    rnn_average = load_rnn_data(data_dir + "merged_rep_average.csv")

    transfer_num, transfer_accumTime, train_x_index, label_scalers2 = load_lable2()

    x1 = get_LinearRegData(rnn_rep, head, tail)
    x2 = get_LinearRegData(rnn_rep2, head, tail)
    x3 = get_LinearRegData(base20_23, head, tail)
    x4 = get_LinearRegData(rnn_average, head, tail)
    my_x = {}
    my_x["1"] = x1
    my_x["2"] = x2
    my_x["3"] = x3
    my_x["4"] = x4

    # ratio1 = find_ratio_of_topn(x1, 10000)
    # ratio2 = find_ratio_of_topn(x2, 10000)

    cut_ratio = np.arange(2, 10) / 10
    MAPE = np.zeros((my_x.__len__(), cut_ratio.size), dtype=float)

    start = time.perf_counter()
    for i, x_now in enumerate(my_x):
        for j, cut in enumerate(cut_ratio):
            average_MAPE = 0
            for k in range(0, 5):
                train_x, train_y, test_x, test_y = divide_dataset(cut, my_x[x_now], edge_labels)
                # train_x, train_y, test_x, test_y = divide_dataset(0.2, rnn_rep[train_x_index,:], transfer_accumTime)
                # train_x = train_x.cpu().detach().numpy()
                # train_y = train_y.cpu().detach().numpy()
                # test_x = test_x.cpu().detach().numpy()
                # test_y = test_y.cpu().detach().numpy()

                batch_size = train_x.shape[0]
                # batch_size = 128
                train_data = TensorDataset(train_x, train_y)
                train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, drop_last=True)

                # 4.torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
                is_cuda = torch.cuda.is_available()
                # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
                if is_cuda:
                    device = torch.device("cuda")
                    # print("GPU is available")
                else:
                    device = torch.device("cpu")
                    # print("GPU not available, CPU used")

                lr = 0.0002
                nn_model = train(train_loader, lr)

                # nn_outputs, real_y, gru_sMAPE = evaluate(nn_model, test_x, test_y, label_scalers)
                nn_outputs, real_y, gru_sMAPE = evaluate(nn_model, test_x, test_y, label_scalers)
                average_MAPE += gru_sMAPE
            MAPE[i][j] = average_MAPE / 5
        print(i, "----->", j)

    end = time.perf_counter()
    print('Running time: %.2f minutes' % ((end - start) / 60.0))
    show_compare_curve(cut_ratio, MAPE)

    # plt.figure(figsize=(14,10))
    # plt.subplot(2,2,1)
    # plt.plot(nn_outputs[:30], "-o", color="g", label="Predicted")
    # plt.plot(real_y[:30], color="b", label="Actual")
    # plt.ylabel('Station transfer num')
    # plt.legend()
    #
    # plt.subplot(2,2,2)
    # plt.plot(nn_outputs[-30:-1], "-o", color="g", label="Predicted")
    # plt.plot(real_y[-30:-1], color="b", label="Actual")
    # plt.ylabel('Station transfer num')
    # plt.legend()
    #
    # plot_y = label_scalers.inverse_transform(edge_labels.reshape(-1,1))
    # lookup = 1
    # plt.subplot(2,2,3)
    # plt.plot(x1[:,lookup], plot_y, "o", color="g", label="similarity")
    # plt.xlabel('two station vector similarity')
    # plt.ylabel('Station transfer num')
    #
    # plt.subplot(2,2,4)
    # plt.plot(x2[:,lookup], plot_y, "o", color="b", label="similarity")
    # plt.xlabel('two station vector similarity')
    # plt.ylabel('Station transfer num')
    # plt.show()

