import numpy as np
import matplotlib.pyplot as plt
import math


class BasicLinearClassifier:
    """
    Linear classified based on centroids for n-dimensional data.
    """

    def __init__(self):
        """
        Initialize parameters.
        """
        
        self.threshold = None  # threshold for class prediction
        self.w = None          # orthogonal vector to threshold 


    def train(self, X, y, weights):
        """
        Train our model given a set of instances, their labels, and their weights.

        Parameters 
        ----------
        X : { numpy.ndarray } of shape (n_samples, n_features)
            Training instances with dtype=np.float32.

        y : { numpy.ndarray } of shape (n_samples,)
            Labels of training instances in {-1, +1}.

        weights : { numpy.ndarray } of shape (n_samples,)
            Weights for each instance

        Returns
        -------
        self : object 
        """

        # print("\ttraining linear classifier")
        # print("\tsum of weights (should be 1): ", np.sum(weights))

        # Sort data into positive and negative classes
        n_features = X.shape[1]
        # training_classe 
        #     Create a dictionary mappiny each training class to a numpy array of tuples
        #     First entry is weight, second entry is training instance
        training_classes = {-1 : (np.empty((0,1)), np.empty((0, n_features))), 
                             1 : (np.empty((0,1)), np.empty((0, n_features)))}  
        for i in range(X.shape[0]):
            c = int(y[i])
            weight_i = weights[i]
            X_i = X[i]
            # print("\t\tX = ", X_i)
            # print("\t\tw = ", weight_i)
            training_classes[c] = (np.vstack((training_classes[c][0], weights[i])),
                                   np.vstack((training_classes[c][1], X[i])))

        c1 = self.__compute_centroid(*training_classes[-1])
        c2 = self.__compute_centroid(*training_classes[1])

        self.__compute_discriminant(c1, c2)

        # plot_data(training_classes[1][1], training_classes[-1][1], c2, c1, "data.png")

        return self

    def predict(self, X):
        """
        Given a list of instances, predict their output. 

        Parameters
        ----------
        X : { numpy.ndarray } of shape (n_samples, n_features)
            Testing instances

        Returns
        -------
        y_pred : { numpy.ndarray } of shape (n_samples,)
            Predictions for each test instance in X
        """

        n_samples = X.shape[0]
        pred = np.zeros((n_samples, 1))
        for i in range(n_samples):
            x = X[i]
            if np.dot(x, self.w) > self.threshold:
                pred[i] = -1
            else:
                pred[i] = 1

        return pred

    def error(self, X, y, weights):
        """
        Given a list of instances, predict their output and calculate WEIGHTED error.

        Parameters
        ----------
        X : { numpy.ndarray } of shape (n_samples, n_features)
            Testing instances
        y : { numpy.ndarray } of shape (n_samples, )
            Testing labels
        weights : { numpy.ndarray } of shape (n_samples, )

        Returns
        -------
        err : { dtype.int }
            Calculated error
        pred : { numpy.ndarray } of shape (n_samples, )
            Predictions
        """

        pred = self.predict(X)
        
        TP = 0
        FN = 0
        TN = 0
        FP = 0

        for i in range(len(y)):
            pred_label = pred[i]
            gt_label = y[i]

            if int(pred_label) == -1:
                if pred_label == gt_label:
                    TN += 1 * weights[i]
                else:
                    FN += 1 * weights[i]
            else:
                if pred_label == gt_label:
                    TP += 1 * weights[i]
                else:
                    FP += 1 * weights[i]

        err = 1 - ( (TP + TN) / (TP + FN + FP + TN) )

        return err, pred

    def __compute_centroid(self, w, X):
        """
        Compute the centroid of a class given the training instances of a class.

        Parameters
        ----------
        X : { numpy.ndarray } of shape (n_samples, n_features)
            Testing instances of a given class
        w : { numpy.ndarray } of shape (n_samples,)
            Weights for each testing instance

        Returns
        -------
        c : { dtype.float32 } 
            Centroid (real number) of class
        """

        c = np.sum(w * X, axis=0) / np.sum(w, axis=0)
        return c

    def __compute_discriminant(self, c1, c2):
        """
        Compute the discriminant given two class centroids.

        Parameters
        ----------
        c1, c2 : { numpy.ndarray } of shape (, n_features)
            Centroids of class 1 and 2

        Returns 
        -------
        threshold : { dtype.float32 } 
            Real number threshold for class assignments
        w : { numpy.ndarray } of shape (, n_feature)
            Orthogonal vector
        """

        self.threshold = np.dot((c1 - c2), (c1 + c2)) / 2
        self.w = c1 - c2
        return self.threshold, self.w


class BoostingClassifier:
    """
    Boosting classifier with algorithm 
    """
    def __init__(self, T=5, A=BasicLinearClassifier):
        """
        Initialize a BoostingClassifier instance.

        Parameters
        ----------
        T : { dtype.int } 
            Number of models to train
        A : { class } 
            Algorithm we'd like to use
        """
        
        self.T = T
        self.A = A

        # create an empty list of models
        self.M = [None for i in range(T + 1)]
        self.alpha = [None for i in range(T + 1)]


    def fit(self, X, y, v=True):
        """ 
        Fit the boosting model.

        Parameters
        ----------
        X : { numpy.ndarray } of shape (n_samples, n_features)
            The input samples with dtype=np.float32.
        
        y : { numpy.ndarray } of shape (n_samples,)
            Target values. By default, the labels will be in {-1, +1}.

        v : { dtype.bool } 
            Whether or not to print verbosely

        Returns
        -------
        self : object
        """

        n_samples = X.shape[0]   # number of samples in training set
        n_features = X.shape[1]  # number of features in each instance

        w = np.empty(shape=(self.T + 1, n_samples))
        w[1] = np.array([1 / n_samples for i in range(n_samples)])  # initialize weights 

        # iterate over T training instances
        for t in range(1, self.T+1):

            if v:
                print("Iteration " + str(t) + ":")

            self.M[t] = self.A().train(X, y, w[t])  # train algorithm A on X, y, with weights w
            err, pred = self.M[t].error(X, y, w[t])    # calculate the errors of this prediction for the training set
            
            if v:
                print("Error = " + str(err))
            
            if err >= 0.5:                       # assert that our error is less than one half
                self.T = t - 1
                break;
            self.alpha[t] = 0.5 * math.log( (1 - err) / (err) )  # calculate alpha (confidence)
            
            if v:
                print("Alpha = " + str(self.alpha[t]))
                f_inc = 1 / (2 * err)
                f_dec = 1 / (2 * (1 - err))
                print("Factor to increase weights = " + str(f_inc))
                print("Factor to decrease weights = " + str(f_dec))

            # iterate over instances and recalculate weights
            if t != self.T:
                # print("\trecalculating weights")
                for i in range(n_samples):
                    pred_label = pred[i]                  # predicted label
                    gt_label = y[i]                       # ground truth label
                    if int(pred_label) == int(gt_label):  # correctly classified case
                        # print("\t\t{} correctly classified as {}".format(X[i], int(pred_label)))
                        # print("\t\t\tResetting weight from {} to {}".format(w[t][i], w[t][i] / (2 * (1 - err)))) 
                        w[t+1][i] = w[t][i] / (2 * (1 - err))
                    else:                                 # incorrectly classified
                        # print("\t\t{} incorrectly classified as {}".format(X[i], int(pred_label)))
                        # print("\t\t\tResetting weight from {} to {}".format(w[t][i], w[t][i] / (2 * err)))
                        w[t+1][i] = w[t][i] / (2 * err)

        return self

    def predict(self, X):
        """ 
        Predict binary class for X.

        Parameters
        ----------
        X : { numpy.ndarray } of shape (n_samples, n_features)

        Returns
        -------
        y_pred : { numpy.ndarray } of shape (n_samples)
        """

        y_pred = np.zeros(shape=(X.shape[0], 1))

        for i in range(1, self.T+1):
            pred_t = self.M[i].predict(X)
            weighted_pred_t = self.alpha[i] * pred_t
            y_pred += weighted_pred_t

        y_pred = np.sign(y_pred)
        return y_pred


def plot_data(pos, neg, p_c, n_c, plot_name):
    """
    Plot data and centroids.

    Only works for 2-dimensional data.

    Parameters 
    ----------
    pos: { numpy.ndarray } of shape (n_positive_samples, n_features)
        Positive instances
    neg: { numpy.ndarray } of shape (n_negative_samples, n_features)
        Negative instances
    p_c: { numpy.ndarray } of shape (1, n_features)
        Positive centroid
    n_c: { numpy.ndarray } of shape (1, n_features)
        Negative centroid
    plot_name : { dtype.str }
        Name of plot
    """

    fig_width = 5
    fig_height = 3
    panel_width = 2.8 / fig_width
    panel_height = 1.9 / fig_height

    plt.figure(figsize=(fig_width, fig_height))
    panel = plt.axes((0.25, 0.25, panel_width, panel_height))

    panel.set_title("data and calculated centroids")
    panel.set_xlabel("x")
    panel.set_ylabel("y")

    # print(pos)
    # print(neg)
    
    pos_x = np.hsplit(pos, 2)[0]
    pos_y = np.hsplit(pos, 2)[1]

    neg_x = np.hsplit(neg, 2)[0]
    neg_y = np.hsplit(neg, 2)[1]

    panel.scatter(pos_x, pos_y, s=0.25, c='blue')
    panel.scatter(neg_x, neg_y, s=0.25, c='orange')

    panel.scatter(p_c[0], p_c[1], s=0.5, marker='X', c='darkblue')
    panel.scatter(n_c[0], n_c[1], s=0.5, marker='X', c='red')

    plt.savefig(plot_name, dpi=600)

