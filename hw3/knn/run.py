import numpy as np
import matplotlib.pyplot as plt
import math
import random


def calc_distances(test_instance, training_data):
    """
    Calculate the distance from each training point of a test instance.

    test_instance: 1 x n numpy array
        testing instance
    training_data: m x n numpy array
        training data
    """

    distances = [i for i in range(training_data.shape[0])]
    for i in range(training_data.shape[0]):
        d = np.linalg.norm(test_instance - training_data[i])
        distances[i] = d
    return distances


def find_accuracy(predicted, actual):
    """
    Calculate the percentage of correct predictions.

    predicted: numpy array
        list of predicted labels
    actual: numpy array
        list of actual labels
    """

    acc = [1 if predicted[i] == actual[i] else 0 for i in range(len(predicted))]
    return sum(acc) / len(acc)


def find_class(k, idx, y_train):
    """
    Find the most-represented class value in the given indices.

    idx: list
        indices of nearest neighbors
    y_train: numpy array
        labels of training data
    """

    labels = [y_train[i] for i in idx[:k]]
    return max(set(labels), key=labels.count)


def find_min_k(k, distances):
    """
    Find the k minimum indices of list of distances.

    k: int
        k for kNN algorithm
    distances: list
        list of distances from test instance to training data
    """

    idx = np.argpartition(distances, k)
    return idx


def partition_data(x_train, y_train, percentage=0.10):
    """
    Given a training set and labels, randomly partition testing set.

    x_train: numpy array
        training instances
    y_train: numpy array
        training labels
    percentage: float
        percentage of data to partition for testing
    """
    
    n = x_train.shape[0]  # number of training samples
    test_size = math.floor(n * percentage)  # number of testing samples
    test_indices = random.sample(range(n), test_size)

    # preallocate training and testing sets
    x_train_partition = np.empty(shape=(n - test_size, x_train.shape[1]))
    y_train_partition = np.empty(shape=(n - test_size,))

    x_test_partition = np.empty(shape=(test_size, x_train.shape[1]))
    y_test_partition = np.empty(shape=(test_size,))

    j = 0
    k = 0
    for i in range(n):  # iterate over all training instances
        if i in test_indices:
            x_test_partition[j] = x_train[i]
            y_test_partition[j] = y_train[i]
            j += 1
        else:
            x_train_partition[k] = x_train[i]
            y_train_partition[k] = y_train[i]
            k += 1

    return x_train_partition, y_train_partition, x_test_partition, y_test_partition


def plot_accuracy(accuracy):
    """
    Plot accuracy.

    accuracy: list
        list of accuracies for various k values
    """

    fig_width = 5
    fig_height = 3
    panel_width = 2.8 / fig_width
    panel_height = 1.9 / fig_height

    plt.figure(figsize=(fig_width, fig_height))
    panel = plt.axes((0.25, 0.25, panel_width, panel_height))

    panel.set_title("accuracy for different values of k")
    panel.set_xlabel("k")
    panel.set_ylabel("accuracy")

    panel.set_xlim(0, len(accuracy) + 1)
    panel.set_ylim(0, 1.1)

    panel.set_xticks([k+1 for k in range(len(accuracy))])

    panel.set_yticks([0, 0.25, 0.5, 0.75, 1])

    panel.plot([k+1 for k in range(len(accuracy))], accuracy,
               linestyle='--', linewidth=0.5, color='black',
               marker='o', markeredgewidth=0, markerfacecolor='red', markersize=1.5)

    plt.savefig("accuracy.png", dpi=600)



def run(Xtrain_file, Ytrain_file, test_data_file, pred_file, 
        k=5, file_handles=True, write_output=True):
    """
    Given training data X and labels Y, produce predictions for training instances.

    Parameters:
        Xtrain_file: string or numpy array
            path to training data or numpy array containing data
        Ytrain_file: string or numpy array
            path to or array of training labels
        test_data_file: string or numpy array
            path to or numpy array of testing instances
        pred_file: string
            path to write output
        k: int
            k parameter for kNN algorithm
        file_handles: boolean
            whether we are given file handles or numpy arrays, default True
        write_output: boolean
            whether to write or return output as numpy array, default True
    """

    ###########################################################################
    #
    # Read in data
    #
    ###########################################################################

    if file_handles:

        # training data
        x_train = np.loadtxt(Xtrain_file, delimiter=',')
        y_train = np.loadtxt(Ytrain_file, delimiter=',')

        # test data
        test_data = np.loadtxt(test_data_file, delimiter=',')

    else:

        x_train = Xtrain_file
        y_train = Ytrain_file

        test_data = test_data_file

    ###########################################################################
    #
    # Produce predictions 
    #
    ###########################################################################

    predictions = list()
    for instance in test_data:
        d = calc_distances(instance, x_train)
        min_indices = find_min_k(k, d)
        predicted_class = find_class(k, min_indices, y_train)
        predictions.append(predicted_class)

    if write_output:
        np.savetxt(pred_file, predictions, fmt="%ld", delimiter=",")
    else:
        return predictions

def main():
    """
    Partition training data and test model.
    """

    x_train = np.loadtxt('data/Xtrain.csv', delimiter=',')
    y_train = np.loadtxt('data/Ytrain.csv', delimiter=',')

    accuracy = [0 for i in range(10)]
    r = 100
    for i in range(r):
        x_train_p, y_train_p, x_test_p, y_test_p = partition_data(x_train, y_train)
        for k in range(1, 11):
            pred = run(x_train_p, y_train_p, x_test_p, None, 
                       k=k, file_handles=False, write_output=False)
            a = find_accuracy(pred, y_test_p)
            accuracy[k - 1] += a
    accuracy = [a / r for a in accuracy]
    plot_accuracy(accuracy)
            


if __name__ == "__main__":
    main()

