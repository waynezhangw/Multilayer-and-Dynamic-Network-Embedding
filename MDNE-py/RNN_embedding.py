# This python file is used to implement dynamic (multi-period) results fusion
# based on one version of Recurrent Neural Networks, Gated Recurrent Unit
# Author: Wenyuan Zhang
# Time: 2020/03/29

import os
import time
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from numpy import zeros, hstack, vstack, float32 as REAL
from torch.autograd import Variable

from tqdm import tqdm_notebook
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

RS = 123  #To maintain reproducibility, you will define a random state variable RS and set it to 123

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

# load edge data
def load_lable():
    df = pd.read_csv(data_dir + "20170405_edgeType5-20.csv", header='infer')
    # show_relation(df.values[:,3], df.values[:,7], df.values[:,3])
    head = df.values[:,1]
    head = head.reshape(-1, 1)
    tail = df.values[:,2]
    tail = tail.reshape(-1, 1)
    label_sc = MinMaxScaler()
    data = label_sc.fit_transform(df.values)
    edge_labels = df.values[:,7]    # read transfer times and cost
    transfer_num = df.values[:,3]
    # edge_labels = 1 - edge_labels
    edge_labels = edge_labels.reshape(-1, 1)
    label_sc.fit(df.iloc[:, 7].values.reshape(-1, 1))
    label_scalers = label_sc
    head, tail, edge_labels, transfer_num = generate_negative_samples(head, tail, edge_labels, transfer_num)
    return head, tail, edge_labels, label_scalers, transfer_num

def generate_negative_samples(head, tail, edge_labels, transfer_num):
    head = head.reshape(-1).astype(int)
    tail = tail.reshape(-1).astype(int)
    edge_labels = edge_labels.reshape(-1)
    transfer_num = transfer_num.reshape(-1)
    nodes = vstack((head, tail)).reshape(-1).tolist()
    nodes = list(set(nodes))
    head_tail_dict = {}
    for i, he in enumerate(head):
        key = str(head[i])+"-"+str(tail[i])
        head_tail_dict[key] = True
    g_head = []
    g_tail = []
    g_labels = []
    g_transfer_num = []
    K = len(nodes)
    for h in head:
        g_head.append(h)
        while True:
            kk = int(np.floor(np.random.rand() * K))
            temp_tail = nodes[kk]
            temp_key = str(h)+"-"+str(temp_tail)
            if temp_key in head_tail_dict:
                continue
            g_tail.append(temp_tail)
            break
        temp_label = -np.random.rand()
        g_labels.append(-1)
        g_transfer_num.append(0.1)
    head_g = hstack((np.array(g_head), head))
    tail_g = hstack((np.array(g_tail), tail))
    edge_labels_g = hstack((np.array(g_labels), edge_labels))
    transfer_num_g = hstack((np.array(g_transfer_num), transfer_num))
    return head_g.reshape(-1, 1), tail_g.reshape(-1, 1), edge_labels_g.reshape(-1, 1), transfer_num_g.reshape(-1, 1)


# 2.load and pre-process the data
def load_rnn_data(data_dir):
    head, tail, labels, label_scalers, transfer_num = load_lable()
    # read sequence data
    look_back = 5
    inputs = np.zeros((10626, look_back, 200))  # 200 is dim of the input features
    for i in range(0, 5):
        base = load_model(data_dir + 'base_' + str(i+1) + '.npy', data_dir + 'index2word_' + str(i+1) + '.json')
        sc = MinMaxScaler()
        data = sc.fit_transform(base)
        inputs[:, i, :] = data
        # inputs = inputs.reshape(-1, look_back, data.shape[1])

    test_portion = int(0.001 * len(head))
    train_x = inputs  # 10626 * 5 * 200
    train_x_index = np.arange(0, train_x.shape[0]).reshape(-1,1)
    train_y = labels


    print("data pre-processing over:")
    print(train_x.shape)
    return head, tail, train_x, train_x_index, train_y, label_scalers, transfer_num

def train(train_loader, label_loader, learn_rate, hidden_dim=200, EPOCHS=200, model_type="GRU"):
    # Setting common hyperparameters
    my_loss = []
    input_dim = next(iter(train_loader))[0].shape[2]
    output_dim = 1
    n_layers = 2
    final_representation = np.zeros((10626,int(hidden_dim)))
    final_rep = torch.from_numpy(final_representation).float().to(device)
    # Instantiating the models
    if model_type == "GRU":
        model = GRUNet(input_dim, hidden_dim, output_dim, n_layers)
    else:
        model = LSTMNet(input_dim, hidden_dim, output_dim, n_layers)
    model.to(device)

    # Defining loss function and optimizer
    # criterion = nn.MSELoss(reduction = 'sum')
    # criterion = nn.MSELoss()
    # criterion = nn.KLDivLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.train()
    print("Starting Training of {} model".format(model_type))
    epoch_times = []
    temp_epoch = 0
    # fetch data
    my_label = 0
    my_head = 0
    my_tail = 0
    for x1, y1, z1 in label_loader:
        my_label = x1
        my_head = y1
        my_tail = z1
        break

    adj = model.my_adj(my_label, my_head, my_tail)
    x = 0
    x_index = 0
    for x1, x2 in train_loader:
        x = x1
        x_index = x2
        break
    # Start training loop
    for epoch in range(1, EPOCHS + 1):
        temp_epoch += 1
        start_time = time.perf_counter()
        h = model.init_hidden(train_batch_size)
        avg_loss = 0.
        counter = 0
        # for x, x_label in train_loader:
        counter += 1
        if model_type == "GRU":
            h = h.data
        else:
            h = tuple([e.data for e in h])
        model.zero_grad()

        out, h = model(x.to(device).float(), h, x_index, my_head, my_tail)
        # loss = criterion(out, my_label.to(device).float())
        loss = model.my_loss(h[1,:,:], adj)
        # loss = model.my_direct_loss(h[1, :, :], my_label, my_head, my_tail)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        # if counter % 20 == 0:
        #     print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter,
        #                                                                                len(train_loader),
        #                                                                                avg_loss / counter))


            # temp_head = (head.cpu().numpy() - 1).reshape(-1)
            # temp_tail = (tail.cpu().numpy() - 1).reshape(-1)
            # meme = int(h.shape[2] / 2)
            # temp_front = h[1, :, 0:meme]
            # temp_back = h[1, :, meme:h.shape[2]]
            # temp1 = final_rep[temp_head, :] + temp_front
            # temp2 = final_rep[temp_tail, :] + temp_back
            # final_rep[temp_head, :] = temp1 / 2
            # final_rep[temp_tail, :] = temp2 / 2

        current_time = time.perf_counter()
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, loss.item()))
        my_loss.append(loss.item())
        if (loss.item() < 100  or epoch == EPOCHS):
            final_rep = h[1,:,:]
            break

        # print("Time Elapsed for Epoch: {} seconds".format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)
        # when fish one epoch, calculate overall loss

        # if ((avg_loss / len(train_loader)) < 1.0e-6):
        #     break
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    temp_final = final_rep.cpu()
    return model, temp_final.detach().numpy(), np.array(my_loss)

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
        self.sigmoid = nn.Sigmoid()
        self.selu = nn.SELU()
        self.tanhshrink = nn.Tanhshrink()
        self.tanh = nn.Tanh()
        self.prelu = nn.PReLU()

    def forward(self, x, h, temp_index, temp_head, temp_tail):
        out, h = self.gru(x, h)

        # my_rep = h[1,:,:]
        # temp_head = (temp_head.cpu().numpy() - 1).reshape(-1)
        # temp_tail = (temp_tail.cpu().numpy() - 1).reshape(-1)
        # big_head = my_rep[temp_head,:]
        # big_tail = my_rep[temp_tail,:]
        # final_big = torch.mul(big_head, big_tail)
        # out = torch.sum(final_big, dim=1)
        # out = self.tanhshrink(out)
        # out = out.reshape(-1,1)

        # meme = int(h.shape[2]/2)
        # temp_front = h[1,:,0:meme]
        # temp_back = h[1,:,meme:h.shape[2]]
        # temp_matrix = torch.mm(temp_front, temp_back.t())
        # key_diagonal = np.arange(0,h.shape[1])
        # out = temp_matrix[key_diagonal, key_diagonal]
        # out = self.relu(out)
        # out = out.reshape(-1,1)

        # temp_noneLinear = out[:, -1]
        # out = self.fc(self.relu(out[:, -1]))
        # out = self.fc(self.sigmoid(temp_noneLinear))
        return out, h

    def my_loss(self, H, adj_mini_batch):
        temp = adj_mini_batch.data.numpy()
        D = torch.diag(torch.from_numpy(temp.sum(1)))
        L = D - adj_mini_batch
        L = L.type(torch.FloatTensor)
        return torch.abs(2 * torch.trace(torch.matmul(torch.matmul(torch.transpose(H, 1, 0), L), H)))

    def my_adj(self, label, head, tail):
        temp_array = np.zeros((10626,10626), dtype=float)
        head_t = head.data.numpy().reshape(-1).astype(int)
        tail_t = tail.data.numpy().reshape(-1).astype(int)
        label_t = label.data.numpy().reshape(-1)
        temp_array[head_t-1, tail_t-1] = label_t
        return torch.from_numpy(temp_array)

    def my_direct_loss(self, H, label, head, tail):
        head_t = (head.data.numpy() - 1).reshape(-1)
        tail_t = (tail.data.numpy() - 1).reshape(-1)
        big_head = H[head_t.astype(int), :]
        big_tail = H[tail_t.astype(int), :]
        final_x = ((big_head - big_tail) ** 2)/200
        final_x = torch.sqrt(torch.sum(final_x, 1))
        final_x = torch.mul(final_x.view(-1,1), label)
        final_x = torch.sum(final_x)
        return final_x

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

    def forward(self, x, h, temp_index, temp_head, temp_tail):
        out, h = self.lstm(x, h)
        my_rep = h[1][1, :, :]
        temp_head = (temp_head.cpu().numpy() - 1).reshape(-1)
        temp_tail = (temp_tail.cpu().numpy() - 1).reshape(-1)
        big_head = my_rep[temp_head, :]
        big_tail = my_rep[temp_tail, :]
        final_big = torch.mul(big_head, big_tail)
        out = torch.sum(final_big, dim=1)
        out = self.relu(out)
        out = out.reshape(-1, 1)
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden

def show_sTNE(data, label):
    # data: sample num*feature num，say station num*embedding dimension
    # label: sample num*class label from K-Means
    tsne = TSNE(random_state=RS)
    X_embedded = tsne.fit_transform(data)
    # sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=label, legend='full', palette=palette)
    num_classes = len(np.unique(label))
    palette = np.array(sns.color_palette("hls", num_classes))
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], lw=0, s=40, c=palette[label.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []
    for i in range(num_classes):
        # Position of each label at median of data points.
        xtext, ytext = np.median(X_embedded[label == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=34)
        txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()])
        txts.append(txt)

    plt.show()
    return f, ax, sc, txts

def get_LinearRegData(station_rep, head, tail):
    head = (head-1).reshape(-1)
    tail = (tail-1).reshape(-1)
    big_head = station_rep[head.astype(int), :]
    big_tail = station_rep[tail.astype(int), :]
    final_x = (np.subtract(big_head, big_tail)**2)/station_rep.shape[1]
    final_x = np.sqrt(np.sum(final_x, axis=1))
    return final_x

def show_relation(x1, x2, x3):
    negative_length = int(x3.shape[0]/2)
    x3[0:negative_length] = -10
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 1, 1)
    plt.plot(x1[0:negative_length], x3[0:negative_length], "o", color="g", label="similarity")
    plt.plot(x1[negative_length:-1], x3[negative_length:-1], "o", color="b", label="similarity")
    plt.ylabel('Station transfer times')

    plt.subplot(2, 1, 2)
    plt.plot(x2[0:negative_length], x3[0:negative_length], "o", color="m", label="similarity")
    plt.plot(x2[negative_length:-1], x3[negative_length:-1], "o", color="b", label="similarity")
    plt.ylabel('Station transfer times')
    plt.show()

def show_loss_epoch(loss):
    x = np.arange(1, loss.size+1)
    plt.plot(x, loss, "-", color="r", label="loss")
    # plt.yscale('log')
    plt.title('model loss')
    plt.xlabel('epoch number')
    plt.ylabel('loss value')
    plt.show()

if __name__ == "__main__":
    # 1.Define the root directory of station vector data and label data
    data_dir = "./data_GRU/"
    files = os.listdir(data_dir)
    print(files)
    head, tail, train_x, train_x_index, train_y, label_scalers, transfer_num = load_rnn_data(data_dir)
    average_x = np.zeros((train_x.shape[0], train_x.shape[2]))
    for i in range(0, train_x.shape[1]):
        average_x += train_x[:, i, :]
    np.savetxt('data_GRU/merged_rep_average.csv', X=average_x / 5, fmt='%f', delimiter=",")

    # 3.To improve the speed of our training, we can process the data in batches so that the model does not need to update its weights as frequently.
    #   The Torch Dataset and DataLoader classes are useful for splitting our data into batches and shuffling them.
    train_batch_size = train_x.shape[0]
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_x_index))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size, drop_last=False)

    label_batch_size = train_y.shape[0]
    label_data = TensorDataset(torch.from_numpy(train_y), torch.from_numpy(head), torch.from_numpy(tail))
    label_loader = DataLoader(label_data, shuffle=True, batch_size=label_batch_size, drop_last=False)

    # 4.torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()
    is_cuda = False
    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    lr = 0.001
    gru_model, merged_rep_gru, my_loss = train(train_loader, label_loader, lr, model_type="GRU")
    show_loss_epoch(my_loss)
    # lstm_model, merged_rep_lstm = train(train_loader, label_loader, lr, model_type="LSTM")
    # merged_rep_gru = merged_rep_lstm

    x1 = get_LinearRegData(merged_rep_gru, head, tail)
    base20_23 = load_model(data_dir + 'base.npy', data_dir + 'index2word.json')
    x2 = get_LinearRegData(base20_23, head, tail)

    show_relation(x1, x2, train_y)

    np.savetxt('data_GRU/merged_rep_gru.csv', X=merged_rep_gru, fmt='%f', delimiter=",")
    estimator = KMeans(n_clusters=5)  # construct cluster
    estimator.fit(merged_rep_gru)  # clustering
    label_pred = estimator.labels_  # get cluster label
    show_sTNE(merged_rep_gru, label_pred)
    np.savetxt('data_GRU/all_stops_label.csv', X=label_pred, fmt='%d', delimiter=",")
