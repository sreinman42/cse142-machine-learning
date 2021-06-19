import numpy as np
import matplotlib.pyplot as plt
import random
import math

def run(Xtrain_file, Ytrain_file, test_data_file, pred_file,
        file_handles=True, T=10, ret=False):
    """
    Train a voted perceptron given data.

    Parameters:
        Xtrain_file: string or np array
            file path to Xtrain csv file (feature vectors)
        Ytrain_file: string or np array
            file path to Ytrain csv file (labels)
        test_data_file: string or np array
            path to test data
        pred_file: string
            file handle to save predictions
        file_handles: boolean
            True if our data is passed as files, False if
            data is passed as numpy lists
        T: int
            number of epochs
        ret: boolean
            whether or not we want to return our prediction
    """

    ###########################################################################
    #
    # Read in data
    # 
    ###########################################################################

    if file_handles:  # data is given as file handles that need to be read
        
        # training data
        x_train = np.loadtxt(Xtrain_file, delimiter=',')
        y_train = np.loadtxt(Ytrain_file, delimiter=',')
        y_train = np.array([y if y == 1 else -1 for y in y_train])

        # testing data
        test_data = np.loadtxt(test_data_file, delimiter=',')

    else:  # data is given as numpy arrays
        x_train = Xtrain_file
        y_train = Ytrain_file

        test_data = test_data_file

    ###########################################################################
    # 
    # Train our model 
    #
    ###########################################################################

    v, c = train(x_train, y_train, T)

    ###########################################################################
    #
    # Test our model
    #
    ###########################################################################

    pred_array = test(test_data, v, c)
    pred_array = [0 if pred_array[i] == -1 else 1 for i in range(pred_array.shape[0])]
    print(pred_array)

    if ret:
        return pred_array
    else:
        np.savetxt(pred_file, pred_array, fmt="%ld", delimiter=",")


def train(x, y, T):
    """
    Given a labeled data set and a number of epochs, train a voted perceptron.

    Parameters:
        x: numpy array or list of lists
            list of input vectors
        y: numpy array or list
            list of labels for each vector
        T: integer
            number of epochs

    Returns:
        v: numpy array
            list of vectors 
        c: numpy array
            list of weights for each vector
    """

    num_features = x.shape[1]
    v = np.zeros(shape=(1, num_features))
    c = np.zeros(1)

    k = 0
    m = y.shape[0]
    
    for t in range(T):  # perform T epochs
        for i in range(m):  # iterate over each training example 

            y_hat = np.sign(np.dot(x[i], v[k]))
            if isinstance(y[i], float):
                y_i = y[i]
            else:
                y_i = y[i][0]
            if y_hat == y_i:
                c[k] = c[k] + 1
            else:
                v = np.vstack((v, v[k] + y[i]*x[i]))
                c = np.vstack((c, 1))
                k += 1

    return v, c


def test(x, v, c):
    """
    Given a list of vectors and their weights, find the outputs of each test vector x

    Parameters:
        x: numpy array
            test instances
        v: numpy array
            list of vectors, should have same length as vectors in x
        c: numpy array
            list of weights for vectors in v, should have some number of rows as v

    Returns:
        pred: numpy array
            list of predictions for each x input
    """

    pred = np.empty(shape=(x.shape[0], 1))  # pre-allocated array for predictions
    for i in range(x.shape[0]):
        s = 0
        x_i = x[i]
        for k in range(c.shape[0]):
            c_k = c[k]
            v_k = v[k]
            s += c_k * np.sign(np.dot(x_i, v_k))
        pred[i] = np.sign(s)

    return pred


def plot_accuracy(accuracy, T):
    """
    Given a list of accuracies, plot them.

    Parameters:
        accuracy: list
            list of accuracies achieved
        T: int
            number of epochs used to attain these accuracies
    """

    fig_width = 5
    fig_height = 3

    panel_width = 2.8 / fig_width
    panel_height = 1.9 / fig_height

    plt.figure(figsize=(fig_width, fig_height))
    panel = plt.axes((0.25, 0.25, panel_width, panel_height))

    panel.set_title("accuracy, T="+str(T))
    panel.set_xlabel("percent of remaining training data")
    panel.set_ylabel("accuracy")

    panel.set_xlim(0, 7)
    panel.set_ylim(0, 1.1)

    panel.set_xticks([1, 2, 3, 4, 5, 6])
    panel.set_xticklabels(["1%", "2%", "5%", "10%", "20%", "100%"])

    panel.set_yticks([0.25, 0.5, 0.75, 1])

    panel.plot([1, 2, 3, 4, 5, 6], accuracy,
               linestyle='--', linewidth=0.5, color='black',
               marker='o', markeredgewidth=0, markerfacecolor='red', markersize=1.5)

    png_name = "accuracy_"+str(T)+".png"
    plt.savefig(png_name, dpi=600)


def main(Xtrain, Ytrain, pred_file, T, testing_percentage):
    """
    Given training data and an output file, partition training data into training and testing set and train.

    Parameters:
        Xtrain_file: string
            Training data file name
        Ytrain_file: string
            Training labels file name
        pred_file: string
            output file name
        T: int
            number of epochs
        testing_percentage: float in [0, 1]
            percentage of data to set aside for validation
    """

    ###########################################################################
    #
    # Parse data
    #
    ###########################################################################

    x_train = np.loadtxt(Xtrain_file, delimiter=',')
    y_train = np.loadtxt(Ytrain_file, delimiter=',')
    y_train = np.array([y if y == 1 else -1 for y in y_train])

    ###########################################################################
    #
    # Partition data into training and test sets
    #
    ###########################################################################

    # split 90% training, 10% testing
    # x_part = np.vsplit(x_train, math.floor(x_train.shape[0] * 0.90))
    # y_part = np.vsplit(y_train, math.floot(x_train.shape[0] * 0.90))

    split = math.ceil(x_train.shape[0] * 0.90)

    x_test = np.array(x_train[split:])
    y_test = np.array(y_train[split:])
    
    x_train = np.array(x_train[:split])
    y_train = np.array(y_train[:split])

    ###########################################################################
    #
    # Train our model with different number of epochs and 
    # different proportion of test set 
    #
    ###########################################################################

    frac = [1]
    for T in [5]:
        
        acc = [0, 0, 0, 0, 0, 0]
        for r in range(1):

            for i in range(len(frac)):

                f = frac[i]
                n = x_train.shape[0]  # number of instances in training set
                training_used = math.floor(n * f)
                used_indices = random.sample(range(n), training_used)  # find random indices to use

                # preallocate numpy arrays for testing and training vectors
                x_train_partition = np.empty(shape=(training_used, x_train.shape[1]))
                y_train_partition = np.empty(shape=(training_used, 1))

                # partition training sets
                for j in range(training_used):
                    x_train_partition[j] = x_train[used_indices[j]]
                    y_train_partition[j] = y_train[used_indices[j]]

                pred_file = "pred_"+str(f)+"_"+str(T)+"_"+str(r)+".txt"
                pred = run(x_train_partition, y_train_partition, x_test, pred_file, 
                           file_handles=False, T=T, ret=False)

                print(pred)

                # compare pred to y_test_partition
                # this_acc = sum([1 if pred[k] == y_test[k] 
                #                 else 0 for k in range(pred.shape[0])]) / pred.shape[0]

                # acc[i] += this_acc

        # acc = [a / 20 for a in acc]
        # plot_accuracy(acc, T)


if __name__ == '__main__':
    Xtrain_file = 'data/Xtrain.csv'
    Ytrain_file = 'data/Ytrain.csv'
    test_data_file = None 
    pred_file = None
    main(Xtrain_file, Ytrain_file, pred_file, 10, 1)

