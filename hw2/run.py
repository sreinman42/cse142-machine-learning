import numpy as np
import matplotlib.pyplot as plt

# Compute centroid
# given a an [N, M] numpy array, 
# produce a [1, M] numpy array whose 
# only row represents the centroid of the given data
def compute_centroid (class_data):

    return np.sum(class_data, axis=0) / class_data.shape[0]


# Compute threshold for linear discriminant function
# given two [1, M] arrays (centroids of a given class)
# produce a scalar value threshold
def compute_threshold (class1, class2):

    return np.dot((class1 - class2), (class1 + class2)) / 2


# Compute the vector orthogonal to the discriminant
def compute_orth (class1, class2):

    return class1 - class2

# Return true if positive class, false otherwise
def disctriminant (w, x, t):

    if np.dot(w, x) <= t:
        return False
    else:
        return True

# Plot data and discriminants
def show_data (c0, c1, c2, w0, w1, w2, out_file):

    fig_width = 3
    fig_height = 3

    panel_width = 2 / fig_width
    panel_height = 2 / fig_height

    plt.figure(figsize=(fig_width, fig_height))
   
    for x, y in c0:
        print(x, y)
        plt.plot(x, y, marker='o', markerfacecolor='green', markersize=1)

    for x, y in c1:
        plt.plot(x, y, marker='o', markerfacecolor='yellow', markersize=1)

    for x, y in c2: 
        plt.plot(x, y, marker='o', markerfacecolor='blue', markersize=1)

    plt.savefig(out_file, dpi=600)



def run (train_input_dir,train_label_dir,test_input_dir,pred_file):

    # train model

    # read in data 
    train_data = np.loadtxt(train_input_dir)
    train_labels = np.loadtxt(train_label_dir)

    # sort input data
    d = train_data.shape[1] 
    td_classes = [np.empty((0, d)) for i in range(0, 3)]
    for i in range(train_data.shape[0]):
        c = int(train_labels[i])
        td_classes[c] = np.vstack((td_classes[c], train_data[i]))

    # find centroids
    td_centroids = [0, 0, 0]
    for i in [0, 1, 2]:
        td_centroids[i] = compute_centroid(td_classes[i])

    # find thresholds
    td_thresholds = {(0, 1): 0, (1, 2): 0, (0, 2): 0}
    for p in td_thresholds.keys():
        td_thresholds[p] = compute_threshold(td_centroids[p[0]], td_centroids[p[1]])

    # find w
    td_w = {(0, 1): 0, (1, 2): 0, (0, 2): 0}
    for p in td_w.keys():
        td_w[p] = compute_orth(td_centroids[p[0]], td_centroids[p[1]])

    # print(td_centroids)
    # print(td_thresholds)
    # print(td_w)

    # show_data(td_classes[0], td_classes[1], td_classes[2], 0, 1, 2, 'test.png')

    # test model

    # load test data
    test_data = np.loadtxt(test_input_dir,skiprows=0)

    n = test_data.shape[0]
    prediction = np.zeros((n, 1))
    for i in range(n):
        x = test_data[i]
        if np.dot(x, td_w[(0, 1)]) > td_thresholds[(0, 1)]:
            if np.dot(x, td_w[(0, 2)]) > td_thresholds[(0, 2)]:
                prediction[i] = 0
            else:
                prediction[i] = 2
        else:
            if np.dot(x, td_w[(1, 2)]) > td_thresholds[(1, 2)]:
                prediction[i] = 1
            else:
                prediction[i] = 2

    # save prediction to prediction file
    np.savetxt(pred_file, prediction, fmt='%1d', delimiter=",")

    
if __name__ == "__main__":
    train_input_dir = 'reference/random_order_data/training2.txt'
    train_label_dir = 'reference/random_order_data/training2_label.txt'
    test_input_dir = 'reference/random_order_data/testing2.txt'
    pred_file = 'result2'
    run(train_input_dir,train_label_dir,test_input_dir,pred_file)
