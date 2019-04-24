
from sklearn import datasets
from sklearn import svm
import pickle
import os
import argparse


def train(gamma=0.001, C=100.0):
    """ Function to train, serialize, and save the Iris model.

    :param gamma: gamma parameter used by the SVM fit method.
    :type gamma: float
    :param C: C parameter used by the SVM fit method.
    :type C: float
    :rtype: None -- function will save trained model to the iris_model/model_files directory

    .. note::
        This code is from: https://scikit-learn.org/stable/tutorial/basic/tutorial.html

    """
    # loading the Iris dataset
    iris = datasets.load_iris()

    # instantiating an SVM model from scikit-learn
    svm_model = svm.SVC(gamma=gamma, C=C)

    # fitting the model
    svm_model.fit(iris.data[:-1], iris.target[:-1])

    # serializing the model and saving it to the /model_files folder
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file = open(os.path.join(dir_path, "model_files", "svc_model.pickle"), 'wb')
    pickle.dump(svm_model, file)
    file.close()


def argument_parser():
    """ This method creates and returns the argument parser used for parsing the cli arguments.

    .. note::
        This function is only used to auto generate the CLI documentation.

    """
    parser = argparse.ArgumentParser(description='Command to train the Iris model.')
    parser.add_argument('-gamma', action="store", dest="gamma", type=float,
                        help='Gamma value used to train the SVM model.')
    parser.add_argument('-c', action="store", dest="c", type=float, help='C value used to train the SVM model.')
    return parser


if __name__ == "__main__":
    parser = argument_parser()
    results = parser.parse_args()

    train(gamma=results.gamma, C=results.c)
