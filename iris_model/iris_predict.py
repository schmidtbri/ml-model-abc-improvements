import os
import pickle
from schema import Schema
from numpy import array

from ml_model_abc import MLModel
from iris_model import __version_info__, __display_name__, __qualified_name__, __description__


class IrisModel(MLModel):
    """ A demonstration of how to use the MLModel base class """
    # accessing the package metadata
    display_name = __display_name__
    qualified_name = __qualified_name__
    description = __description__
    major_version = __version_info__[0]
    minor_version = __version_info__[1]

    # stating the input schema of the model as a Schema object
    input_schema = Schema({'sepal_length': float,
                           'sepal_width': float,
                           'petal_length': float,
                           'petal_width': float})

    # stating the output schema of the model as a Schema object
    output_schema = Schema({'species': str})

    def __init__(self):
        """ Class constructor that loads and deserializes the iris model parameters.

        .. note::
            The trained model parameters are loaded from the "model_files" directory.

        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file = open(os.path.join(dir_path, "model_files", "svc_model.pickle"), 'rb')
        self._svm_model = pickle.load(file)
        file.close()

    def predict(self, data):
        """ Method to make a prediction with the Iris model.

        :param data: Data for making a prediction with the Iris model. Object must meet requirements of the input schema.
        :type data: dict
        :rtype: dict -- The result of the prediction, the output object will meet the requirements of the output schema.

        """
        # calling the super method to validate against the input_schema
        super().predict(data=data)

        # converting the incoming dictionary into a numpy array that can be accepted by the scikit-learn model
        X = array([data["sepal_length"], data["sepal_width"], data["petal_length"], data["petal_width"]]).reshape(1, -1)

        # making the prediction and extracting the result from the array
        y_hat = int(self._svm_model.predict(X)[0])

        # converting the prediction into a string that will match the output schema of the model
        # this list will map the output of the scikit-learn model to the output string expected by the schema
        targets = ['setosa', 'versicolor', 'virginica']

        # this hides the actual output of the model, which is just a number, it will ensure that any user of the
        # model receives output that is easily interpretable, in this case the output will be the species name
        species = targets[y_hat]

        return {"species": species}
