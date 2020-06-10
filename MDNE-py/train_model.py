# This python file is used to train the Multiplex Network Embedding model
# This file is also based on the original MNE's train_model.py
# Author: Wenyuan Zhang
# Time: 2020/02/25

from MNE import *
import sys
import time
import datetime

now = datetime.datetime.now()
print ("start time:")
print (now.strftime("%Y-%m-%d %H:%M:%S"))

start = time.perf_counter()
file_name = sys.argv[1]              # data/20170405_edgeType5-8.csv
file_transfer_name = sys.argv[2]     # data/20170405_transferCount5-8.csv

# file_name = 'data/Vickers-Chan-7thGraders_multiplex.edges'
edge_data_by_type, all_edges, all_nodes = load_network_data(file_name)
edge_data_by_type1, all_edges1, all_nodes1 = load_network_data(file_transfer_name)
model = train_model(edge_data_by_type, True, edge_data_by_type1)
# print(model)
# save_model(model, 'model')
save_model_base(model, 'model')

# test_model = load_model('model') # wayne 暂时不执行

end = time.perf_counter()
print('Running time: %.2f minutes'%((end-start)/60.0))

now = datetime.datetime.now()
print ("end time:")
print (now.strftime("%Y-%m-%d %H:%M:%S"))
