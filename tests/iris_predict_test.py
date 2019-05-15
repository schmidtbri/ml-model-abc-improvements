import os
import sys
import unittest
from sklearn import svm
from schema import SchemaError
import json

# this adds the project root to the PYTHONPATH if its not already there, it makes it easier to run the unit tests
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iris_model.iris_predict import IrisModel


class IrisModelPredictTests(unittest.TestCase):
    def test1(self):
        """ testing the __init__() method """
        # arrange, act
        model = IrisModel()

        # assert
        self.assertTrue(type(model._svm_model) is svm.SVC)

    def test2(self):
        """ testing the input schema with wrong data """
        # arrange
        data = {'name': 'Sue', 'age': '28', 'gender': 'Squid'}

        # act
        exception_raised = False
        try:
            validated_data = IrisModel.input_schema.validate(data)
        except SchemaError as e:
            exception_raised = True

        # assert
        self.assertTrue(exception_raised)

    def test3(self):
        """ testing the input schema with correct data """
        # arrange
        data = {'sepal_length': 1.0,
                'sepal_width': 1.0,
                'petal_length': 1.0,
                'petal_width': 1.0}

        # act
        exception_raised = False
        try:
            validated_data = IrisModel.input_schema.validate(data)
        except SchemaError as e:
            exception_raised = True

        # assert
        self.assertFalse(exception_raised)

    def test4(self):
        """ testing the output schema with incorrect data """
        # arrange
        data = {'species': 1.0}

        # act
        exception_raised = False
        try:
            validated_data = IrisModel.output_schema.validate(data)
        except SchemaError as e:
            exception_raised = True

        # assert
        self.assertTrue(exception_raised)

    def test5(self):
        """ testing the output schema with correct data """
        # arrange
        data = {'species': 'setosa'}

        # act
        exception_raised = False
        try:
            validated_data = IrisModel.output_schema.validate(data)
        except SchemaError as e:
            exception_raised = True

        # assert
        self.assertFalse(exception_raised)

    def test6(self):
        """ testing the predict() method throws schema exception when given bad data """
        # arrange
        model = IrisModel()

        # act
        exception_raised = False
        try:
            prediction = model.predict({'name': 'Sue', 'age': '28', 'gender': 'Squid'})
        except SchemaError as e:
            exception_raised = True

        # assert
        self.assertTrue(exception_raised)

    def test7(self):
        """ testing the predict() method with good data"""
        # arrange
        model = IrisModel()

        # act
        prediction = model.predict(data={'sepal_length': 1.0,
                                         'sepal_width': 1.0,
                                         'petal_length': 1.0,
                                         'petal_width': 1.0})

        exception_raised = False
        try:
            IrisModel.output_schema.validate(prediction)
        except SchemaError as e:
            exception_raised = True

        # assert
        self.assertFalse(exception_raised)
        self.assertTrue(type(prediction) is dict)
        self.assertTrue(prediction["species"] == 'setosa')
        self.assertFalse(exception_raised)

    def test8(self):
        """ testing JSON schema generation """
        # arrange
        model = IrisModel()

        # act
        json_schema = model.output_schema.json_schema("https://example.com/my-schema.json")

        # assert
        print(json_schema)
        self.assertTrue(type(json_schema) is dict)

    def test9(self):
        """ testing the properties of the model object """
        # arrange
        model = IrisModel()

        # act
        name = model.name
        qualified_name = model.qualified_name
        description = model.description
        major_version = model.major_version
        minor_version = model.minor_version

        # assert
        self.assertTrue(name == "Iris Model")
        self.assertTrue(qualified_name == "iris_model")
        self.assertTrue(description == "A machine learning model for predicting the species of a flower based on its measurements.")
        self.assertTrue(major_version == 0)
        self.assertTrue(minor_version == 1)


if __name__ == '__main__':
    unittest.main()
