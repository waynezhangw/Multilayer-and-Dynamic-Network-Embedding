# Multilayer-and-Dynamic-Network-Embedding
For the project that used graph embedding to implement multi-layer and dynamic transit network representation and analysis

# 1.MDNE-cpp
This project is for data pre-procssing and data preparation for MDNE-py.

Input:
    (1) trip daily records (up to 3.6 million per day, 15 fileds per row, "20170405-Trip-statistics-2CN-duration.csv")
    (2) Shenzhen station records (10626 stations and their related coordinates, "merge_BusStation改进dbscan加地铁到前面.csv")

Output:
    (1) regular edgeType in a specific time window (8 fields per row, "20170405_edgeType5-20.csv")
        8 fields: edgeType-headID-tailID-ODoccurenceCount-averageODTime-ODdurationOccurCount-averageDurationTime-attractiveness
        used for node embedding in regular commute behavior of transit network
    (2) transfer edgeType in a specific time window (5 fields per row, "20170405_transferCount5-20.csv")
        5 fields: transfer_type-transfer_headID-transfer_tailID-transfer_occurCount-defaultZero
        used for node embedding in transfer behavior of transit network
    (3) count the going out num from current station in a specific time window (2 fields, "20170405_countTransfer5-20.csv")
        2 fileds: (stationID)-theNumStartFromCurrentID-countAccumulateTimeCost
        used in GRU model

# 2.MDNE-py
This project is to get multi-layer embedding results and dynamic-layer embedding results

    (1) train_model.py + biased_Random_walk.py + MNE.py is used to get multi-layer embedding results
        input:
            the output(1) and output(2) of the MDNE-cpp
        output:
            embedding results (10626x200) and its related index
    (2) make_cluster.py is used to visualize clustering results (k-means)
        input:
            the output of the MDNE-py(1)
        output:
            clustering label (10626x1)
    (3) RNN_embedding.py is used to combine the multi-period embedding results to one single result
        input:
            multiple 10626x200
        output:
            one single 10626x200
    (4) pytorch_nn.py is used to predict the transfer num between stations
        input:
            the output of the MDNE-py(3)
        output:
            the predicted results and its accuracy      

# Acknowledgment
We built the training framework based on the original MNE model and adapted that to our transit network embedding. We used the algorithm from LINE, Node2Vec, and idea from MNE.

