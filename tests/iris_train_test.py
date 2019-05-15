import os
import sys
import unittest
from argparse import ArgumentParser

# this adds the project root to the PYTHONPATH if its not already there, it makes it easier to run the unit tests
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iris_model.iris_train import train, argument_parser


class IrisModelTrainTest(unittest.TestCase):
    def test1(self):
        """ testing the train() function with no hyperparameters """
        # arrange, act
        exception_raised = False
        try:
            result = train()
        except Exception as e:
            exception_raised = True

        # assert
        self.assertFalse(exception_raised)

    def test2(self):
        """ testing the train() function with hyperparameter c only"""
        # arrange, act
        exception_raised = False
        try:
            result = train(c=100.0)
        except Exception as e:
            exception_raised = True

        # assert
        self.assertFalse(exception_raised)

    def test3(self):
        """ testing the train() function with hyperparameter gamma only """
        # arrange, act
        exception_raised = False
        try:
            result = train(gamma=0.001)
        except Exception as e:
            exception_raised = True

        # assert
        self.assertFalse(exception_raised)

    def test4(self):
        """ testing the train() function with all hyperparameters """
        # arrange, act
        exception_raised = False
        try:
            result = train(gamma=0.001, c=100.0)
        except Exception as e:
            exception_raised = True

        # assert
        self.assertFalse(exception_raised)

    def test5(self):
        """ testing the  """
        # arrange, act
        result = argument_parser()

        # assert
        self.assertTrue(type(result) is ArgumentParser)


if __name__ == '__main__':
    unittest.main()
