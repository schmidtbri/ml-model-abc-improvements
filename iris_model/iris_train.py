import sys
import os
import pickle
import argparse
import traceback
from sklearn import datasets
from sklearn import svm


def train(gamma=0.001, c=100.0):
    """ Function to train, serialize, and save the Iris model.

    :param gamma: gamma parameter used by the SVM fit method.
    :type gamma: float
    :param c: c parameter used by the SVM fit method.
    :type c: float
    :rtype: None -- function will save trained model to the iris_model/model_files directory

    .. note::
        This code is from: https://scikit-learn.org/stable/tutorial/basic/tutorial.html

    """
    # loading the Iris dataset
    iris = datasets.load_iris()

    # instantiating an SVM model from scikit-learn
    svm_model = svm.SVC(gamma=gamma, C=c)

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

    try:
        # we need these if else statements to handle hyperparameters that are not provided when the cli is called
        if results.gamma is None and results.c is None:
            train()
        elif results.gamma is not None and results.c is None:
            train(gamma=results.gamma)
        elif results.gamma is None and results.c is not None:
            train(c=results.c)
        else:
            train(gamma=results.gamma, c=results.c)
    except Exception as e:
        # printing the error to the screen
        traceback.print_exc()
        # returning error code
        sys.exit(os.EX_SOFTWARE)

    # returning code 0
    sys.exit(os.EX_OK)

