from abc import ABC, abstractmethod


class MLModel(ABC):
    """ An abstract base class for ML model prediction code  """
    @property
    @abstractmethod
    def name(self):
        """ This abstract property returns a display name for the model.

        .. note::
            This is a name for the model that looks good in user interfaces.

        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def qualified_name(self):
        """ This abstract property returns the qualified name of the model.

        .. note::
            A qualified name is an unambiguous identifier for the model. It should be possible to embed it in an URL.

        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def description(self):
        """ This abstract property returns a description of the model. """
        raise NotImplementedError()

    @property
    def major_version(self):
        """ This abstract property returns the model's major version as a string. """
        raise NotImplementedError()

    @property
    def minor_version(self):
        """ This abstract property returns the model's minor version as a string. """
        raise NotImplementedError()

    @property
    @abstractmethod
    def input_schema(self):
        """ This abstract property returns the schema that is accepted by the predict() method. """
        raise NotImplementedError()

    @property
    @abstractmethod
    def output_schema(self):
        """ This abstract property returns the schema that is returned by the predict() method. """
        raise NotImplementedError()

    @abstractmethod
    def __init__(self):
        """ This method holds any deserialization and initialization code for the model. """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, data):
        """ Method to make a prediction with the model.

        :param data: data used by the model for making a prediction
        :type data: object --  can be any python type
        :rtype: python object -- can be any python type

        .. note::
            This method can be used to validate the data parameter in the predict() method implementation in the
            derived class.

        """
        self.input_schema.validate(data)


class MLModelException(Exception):
    """ Exception type used to raise exceptions within MLModel derived classes """
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)
