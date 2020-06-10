# This python file is used to make clustering by using trained embedding results
# Author: Wenyuan Zhang
# Time: 2020/03/26

from MNE import *
import time
import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
import matplotlib.patheffects as PathEffects
from gap_statistic import OptimalK

from sklearn.manifold import TSNE
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

RS = 123  #To maintain reproducibility, you will define a random state variable RS and set it to 123

k_max = 40

def show_kmeans_results(data):
    sum_squared_distances = []
    score = []
    label_pred = dict()
    for k in range(2,k_max):
        estimator = KMeans(n_clusters=k)  # construct cluster
        estimator.fit(data)  # clustering
        label_pred[k] = estimator.labels_  # get the label of the cluster
        sum_squared_distances.append(estimator.inertia_)
        score.append(silhouette_score(data, estimator.labels_, metric='euclidean'))
        print("k = ",k)

    plt.plot(range(2, k_max), sum_squared_distances, marker='o')
    plt.xlabel('K_value')
    plt.ylabel("sum_squared_distances")
    plt.show()

    plt.xlabel('k')
    plt.ylabel("Coefficient")
    plt.plot(range(2, k_max), score, 'o-')
    plt.show()
    return label_pred

def show_gap_stat(data):
    print("begin")
    optimalK = OptimalK(parallel_backend='joblib')
    n_clusters = optimalK(data, cluster_array=np.arange(1, k_max))
    print('Optimal clusters: ', n_clusters)

    optimalK.gap_df.head()
    plt.plot(optimalK.gap_df.n_clusters, optimalK.gap_df.gap_value, linewidth=3)
    plt.scatter(optimalK.gap_df[optimalK.gap_df.n_clusters == n_clusters].n_clusters,
                optimalK.gap_df[optimalK.gap_df.n_clusters == n_clusters].gap_value, s=250, c='r')
    plt.grid(True)
    plt.xlabel('Cluster Count')
    plt.ylabel('Gap Value')
    plt.title('Gap Values by Cluster Count')
    plt.show()

    return n_clusters

def show_sTNE(data, label):
    '''
    :param data: num of samples*num of features，如num of stations*embedding dimension
    :param label: the result (samples*label names) from K-Means
    :return: never mind
    '''
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

def make_10626_label(label, index2word):
    num_of_stops = 10626
    output = [-1]*num_of_stops
    for i, value in enumerate(label):
        output[int(index2word[i])-1] = value

    np.savetxt('model/all_stops_label.csv', X=output, fmt='%d', delimiter=",")

def read_LINE_vec(LINE_folder_name):
    num_of_stops = 10626
    line_dim = 200
    vec_line = zeros((num_of_stops, line_dim), dtype=REAL)
    with open(LINE_folder_name, 'r') as f:
        for pos, line in enumerate(f):
            if (pos==0):
                continue
            words = line.split(' ')
            temp_words = np.array(words[1:-1])
            vec_line[int(words[0])-1] = temp_words

    return vec_line     # 10626*200

def read_LINE_vec_as_base(LINE_folder_name):
    num_of_stops = 10626
    line_dim = 200
    vec_line = zeros((num_of_stops, line_dim), dtype=REAL)
    index2word = list()
    with open(LINE_folder_name, 'r') as f:
        for pos, line in enumerate(f):
            if (pos==0):
                continue
            words = line.split(' ')
            temp_words = np.array(words[1:-1])
            index2word.append(int(words[0]))
            vec_line[pos-1] = temp_words

    base = vec_line[:len(index2word),:]
    return base, index2word

def read_DeepWalk_vec_as_base(DeepWalk_folder_name):
    num_of_stops = 10626
    line_dim = 200
    vec_line = zeros((num_of_stops, line_dim), dtype=REAL)
    index2word = list()
    with open(DeepWalk_folder_name, 'r') as f:
        for pos, line in enumerate(f):
            if (pos==0):
                continue
            words = line.split(' ')
            temp_words = np.array(words[1:-1])
            temp1 = words[-1][:-1]
            temp_words = np.append(temp_words, temp1)
            index2word.append(int(words[0]))
            vec_line[pos-1] = temp_words

    base = vec_line[:len(index2word),:]
    return base, index2word

def concatenate_vectors(base, index2word, LINE_vec):
    num_of_stops = 10626
    line_dim = 200
    vec_line = zeros((num_of_stops, line_dim), dtype=REAL)
    for i, value in enumerate(base):
        vec_line[int(index2word[i])-1] = value
    base = hstack((vec_line, LINE_vec))
    index2word = list(range(1,10626+1))
    return base, index2word

if __name__ == "__main__":
    start = time.perf_counter()

    # 1.read representation result of the MNE
    # trained_model = load_model('model')
    # base = trained_model['base']
    # index2word = trained_model['index2word']

    # 2. read representation result of other method(LINE)
    # base, index2word = read_LINE_vec_as_base('data/LINE_vec2.txt')
    base, index2word = read_DeepWalk_vec_as_base('data/dpwk_vec.txt')

    # 3.to combine the vector from LINE model 1st order with MNE
    # line_model = read_LINE_vec('data/vec.txt')
    # combined_base,index2word = concatenate_vectors(base, index2word, line_model)

    # 4.to see clustering class number
    # label_pred = show_kmeans_results(base)
    # show_gap_stat(base)

    # 5.to draw sTNE and output the label
    estimator = KMeans(n_clusters=13)  # construct cluster
    estimator.fit(base)  # clustering
    label_pred = estimator.labels_  # get the label of the cluster
    show_sTNE(base, label_pred)
    make_10626_label(label_pred, index2word)

    # ------------------------------------------------------------
    end = time.perf_counter()
    print('Running time: %.2f minutes' % ((end - start) / 60.0))
    now = datetime.datetime.now()
    print("current time of finishing the program: ")
    print(now.strftime("%Y-%m-%d %H:%M:%S"))