Title: Improving the MLModel Base Class
Date: 2019-06-12 09:21
Category: Blog
Slug: improving-the-mlmodel-base-class
Authors: Brian Schmidt
Summary: In the previous blog post in this series I showed an object oriented design for a base class that does Machine Learning model prediction. The design of the base class was intentionally very simple so that I could show a simple example of how to use the base class with a scikit-learn model. I showed an easy way to publish schema metadata about the model inputs and outputs, and how to write model deserialization code so that it is hidden from the users of the model. I also showed how to hide the implementation details of the model by translating the user's input to the model's input so that the user of the model doesn't have to know how to use pandas or numpy. In this blog post I will continue to make improvements to the MLModel class and the example that I used in the previous post.

This blog post continues with the ideas developed in the previous post
in this series.

All of the code shown in this post can be found in [this Github repository](https://github.com/schmidtbri/ml-model-abc-improvements).

In the previous blog post in this series I showed an object oriented
design for a base class that does Machine Learning model prediction. The
design of the base class was intentionally very simple so that I could
show a simple example of how to use the base class with a scikit-learn
model. I showed an easy way to publish schema metadata about the model
inputs and outputs, and how to write model deserialization code so that
it is hidden from the users of the model. I also showed how to hide the
implementation details of the model by translating the user's input to
the model's input so that the user of the model doesn't have to know how
to use pandas or numpy. In this blog post I will continue to make
improvements to the MLModel class and the example that I used in the
previous post.

In this blog post I will make the iris example code from the previous
post into a full python package with many features that will make the
iris model easier to install and use from other python packages. I will
also continue to improve the MLModel base class. In general, I want to
show how to make ML code easier to install and use.

When I was doing research for this blog post I found a great [blog post](https://towardsdatascience.com/building-package-for-machine-learning-project-in-python-3fc16f541693)
by [Mateusz Bednarski](https://towardsdatascience.com/@mbednarski)
showing how to build machine learning models as python packages. There
are some similarities between what I will show here and that blog post,
however, this post focuses more on the deployment of ML models into
production systems, whereas Mateusz'z post focuses on packaging the
training code.

This blog post assumes that you have some experience with Python. I will
be referencing resources for learning the tools that I will be using in
the blog post.

## Making the Iris Model into a Python Package

Another improvement that we can make to the example code is to make it
into a full-fledged Python package. This makes it easier to use and
install in other projects. The goal here is to treat ML models as just
another python package, this makes it possible to leverage all of the
tools that Python has for packaging and reusing code. A good guide for
structuring python packages can be found
[here](https://python-packaging.readthedocs.io/en/latest/#).

An common pattern that can be seen in ML code is that it is almost
always hard to use and deploy. This is something that teams that do
machine learning know very well, since the code written by a Data
Scientist almost always needs to be rewritten by a software engineer
before it is possible to deploy it into production systems. Luckily, we
have a lot of tools to make the transition from experimental model to
production model a smoother process. In this section I will show a few
simple steps that will make the example model from [the last blog
post]({filename}/articles/a-simple-ml-model-base-class.md)
into an installable Python package. To accomplish this, we will add
version information to the package, add a command line interface to the
training script, add Sphinx documentation, and add a setup.py file to
the project. As an additional touch, we will automate the documentation
process for the interface of the ML model.

First of all we need to reorganize the code in the project a little bit:

```
- project_root
    - docs (a folder, package documentation will goes in here)
    - iris_model (a folder, iris package code will goes in here)
    - model_files (a folder, the model files go in here)
        - __init__.py (this file is for the python package)
        - iris_predict.py (the prediction code goes here)
        - iris_train.py ( the training script goes here)
    - tests (a folder, unit tests for iris_model package go here)
    - ml_model_abc.py (the MLModel base class goes here)
    - requirements.txt
    - setup.py (the package installation script goes here)
```

A lot of this code is shared with the [previous blog
post]({filename}/articles/a-simple-ml-model-base-class.md),
but it is reorganized here to make it possible to have an ML model that
is can be installed as a Python package.

## Adding Package Versioning

Python packages are usually versioned using [semantic
versioning](https://semver.org/). Software packages that
use semantic versioning must declare a public API. This is complicated
when we want to do versioning of ML models because we have two APIs: the
API for making model predictions and the API for training the model. We
can deal with this complexity by tying the different components of the
semantic version of the package to the prediction API and the training
API of the package.

I chose to version the prediction API of the model using the major and
minor version components of the semantic versioning standard. The
reasoning for this is that a lot of users are affected by changes in the
prediction API, but not as many users are affected by changes in the
training API. This is because ML models are usually used by many people
but trained by a few experts. The patch number of the version can be
used to version changes to the training API.

As an example, whenever the ML model prediction API changes in a
backward-incompatible way the major version number will go up, and
whenever it changes in a backwards-compatible way the minor version will
go up. This approach ensures that any user of the ML model package will
know how changes in the prediction API will affect them when they
install the package. A simple way to understand when to increase the
major or minor version numbers is to do so when the input and output
schemas of the model change. Lastly, any changes to the model training
API will cause the patch version number to go up.

A [common approach](https://packaging.python.org/guides/single-sourcing-package-version/)
for storing version information in a python package is to put a
"\_\_verison\_\_" property into the \_\_init\_\_.py module in the root
of the package:

```python
__version_info__ = (0, 1, 0)
__version__ = ".".join([str(n) for n in __version_info__])
```

The code above can be found
[here](https://github.com/schmidtbri/ml-model-abc-improvements/blob/master/iris_model/__init__.py#L1-L2).

I like to think of an ML model as a software component like any other,
the only difference being that an ML model is statistically significant.
Of course, being statistically significant adds a lot of complexity, but
at the end of the day ML models are just code that can be managed just
like any other piece of code. In this section we can see how to take a
step in that direction by attaching version information to the IrisModel
package.

Although semantic versioning is not designed to be used for versioning
models, we can apply it here to version model code and gloss over the
more complicated aspects of ML models. For example, we can't use
semantic versioning to version model parameters since they are not part
of the codebase and don't have an API. This is a problem that I will
tackle in another blog post.

## Adding a CLI interface to the Training Script

When building ML models, the training code is often written in jupyter
notebooks, while there are ways to automate the training process with
notebooks it's a lot easier to do it through the command line. To do
this we will add a simple command line interface to the Iris model
training script. We will create the interface using the
[argparse](https://docs.python.org/3/library/argparse.html)
package and then create a function that calls the train() function when
the iris\_train.py script is called from the command line.

To create the argparse ArgumentParse object we create a dedicated
function (the reason for this will be explained below):

```python
def argument_parser():
    parser = argparse.ArgumentParser(description='Command to train the Iris model.')
    parser.add_argument('-gamma', action="store", dest="gamma", type=float, help='Gamma value used to train the SVM model.')
    parser.add_argument('-c', action="store", dest="c", type=float, help='C value used to train the SVM model.')
    return parser
```

The code above can be found
[here](https://github.com/schmidtbri/ml-model-abc-improvements/blob/master/iris_model/iris_train.py#L39-L50).

To call the train() function from the command line, I created a new
function called main(). The function gets a parser object, parses the
incoming parameters, and calls the train() function:

```python
def main():
    parser = argument_parser()
    results = parser.parse_args()
    try:
        if results.gamma is None and results.c is None:
            train()
        elif results.gamma is not None and results.c is None:
            train(gamma=results.gamma)
        elif results.gamma is None and results.c is not None:
            train(c=results.c)
        else:
            train(gamma=results.gamma, c=results.c)
    except Exception as e:
        traceback.print_exc()
        sys.exit(os.EX_SOFTWARE)
    sys.exit(os.EX_OK)
```

The code above can be found
[here](https://github.com/schmidtbri/ml-model-abc-improvements/blob/master/iris_model/iris_train.py#L53-L75).

The reason for adding the main() function to wrap the train() function
is so that the main() function can be registered as an entry point when
the iris\_model package is installed. The main() function also handles
parsing the command line arguments, calls the train() function, handles
exceptions and returns the success or error code to the operating system
when the training process is done. Another benefit of this approach is
that train() function can still be imported into other code and called
as a function, but now it also has a CLI interface.

## Adding Sphinx Documentation

One of the great parts of working in the Python ecosystem is the Sphinx
package, which is used for creating documentation from source files.
There are a lot of
[great](https://pythonhosted.org/an_example_pypi_project/sphinx.html)
[guides](https://www.sphinx-doc.org/en/1.5/tutorial.html)
for documenting your package using Sphinx, so I won't go through it
again here. For this blog post, I followed these guides to create a
simple documentation page and [hosted it on Github
pages](https://schmidtbri.github.io/ml-model-abc-improvements/).
Adding documentation is a simple process and it is done by almost all
Python packages that have more than a few users. After putting together
the basic documentation, I followed a few simple extra steps to fully
automate the creation of the documentation for the model.

First of all, I added documentation strings to all classes and methods
in the iris\_model package where it made sense. [Here is an
example](https://github.com/schmidtbri/ml-model-abc-improvements/blob/master/iris_model/iris_predict.py#L41-L47)
of how I documented the predict() method using the docstring in the .py
file. The docstring is formatted so that it can be automatically built
by the sphinx autodoc extension. This extension makes it easy to extract
docstrings from python packages and modules and build documentation. A
good guide for using the autodoc extension can be found
[here](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html).

However, one problem with using the MLModel base class for writing code
is that the predict() method of a class that inherits from MLModel only
accepts a single parameter called "data" as input. This makes it hard to
document the input schema of the model through autodoc since the data
structure accepted by the model for prediction can't be easily described
in the docstring. The same problem happens when we try to document the
return type of the predict() method. Luckily, we can automatically
extract the JSON Schema representation of the input and output schemas
of the model. In order to leverage this, I used the
[sphinx-jsonschema](https://sphinx-jsonschema.readthedocs.io/en/latest/)
extension to automatically add the schema information to a documentation
page. The process for adding it is simple, I just had to add this code
to an .rst file:

```
.. jsonschema:: ../build/input_schema.json
```

The code above can be found
[here](https://github.com/schmidtbri/ml-model-abc-improvements/blob/master/docs/source/iris_predict.rst).

The only problem is that the input and output json schema strings are
not saved to disk for the jsonschema extension to access, but are
available from an instance IrisModel class. To fix this, I added [this
code](https://github.com/schmidtbri/ml-model-abc-improvements/blob/00d5e558f9af7571d824d597107412ed86681e8b/docs/source/conf.py#L29-L39)
to the conf.py file that creates the Sphinx documentation. The code
instantiates an IrisModel object, extracts the JSON Schema strings, and
saves it to a location that can then be read by the Sphinx documentation
generator. The documentation that is generated can be seen
[here](https://schmidtbri.github.io/ml-model-abc-improvements/iris_predict.html#iris-model-input-schema)
and
[here](https://schmidtbri.github.io/ml-model-abc-improvements//iris_predict.html#iris-model-output-schema).

Since we are using the argparse library for creating the CLI interface
for the training script, we can use the
[sphinxarg.ext](https://sphinx-argparse.readthedocs.io/en/stable/index.html)
Sphinx extension to automatically generate the documentation. This was
as easy as adding this code to the .rst file that describes the training
script:

```
.. argparse::
    :module: iris_model.iris_train
    :func: argument_parser
    :prog: iris_train
```

The code above can be found
[here](https://github.com/schmidtbri/ml-model-abc-improvements/blob/master/docs/source/iris_train.rst).

The sphixarg.ext extension then goes to the iris\_train module and calls
the argument parser function, which returns an instance of a
ArgumentParser object, which is then used to generate the documentation.
The results can be seen in the documentation
[here](https://schmidtbri.github.io/ml-model-abc-improvements//iris_train.html#iris-training-code-cli-documentation).

This section shows how it is possible to write the code of an ML model
in such a way that the documentation can be created automatically.
Exposing the input and output schemas of the model as JSON schema
strings makes it possible for a Data Scientist to communicate the
requirements of the model clearly to the end user of the model. At the
same time, by exposing the hyperparameters of the training script as
command line options, its becomes possible to automatically document the
training process. By writing the ML model code in a certain way, it
makes it possible for any changes to the code to be documented
automatically whenever the documentation is generated.

## Adding a setup.py File

Now that we have the ML model code structured as a Python package,
versioned, and documented, we'll add a
[setup.py](https://github.com/schmidtbri/ml-model-abc-improvements/blob/master/setup.py)
file to the project folder. The setup.py file is used by the setuptools
package to install python packages and makes the ML model easily
installable in a virtual environment. A great guide for writing the
setup.py file for your package can be found
[here](https://github.com/kennethreitz/setup.py).

In the iris\_model package setup.py file, most of the fields are very
easy to understand and they are better explained in other guides. In
this blog post, I'll focus on the sections of the setup.py file that had
to be specifically modified for the ML model package. First of all, we
want to point at the folder that contains the iris\_model package, we
can do this with this line in the setup.py file:

```python
packages=["iris_model"],
```

The code above can be found
[here](https://github.com/schmidtbri/ml-model-abc-improvements/blob/master/setup.py#L20).

Next, we need to make sure that the ml\_model\_abc.py Python module is
installed along with the iris\_model package. In the future, it would be
better to take this code and put it into another Python package that the
iris\_model package would depend on, but for now we just need this line
of code:


```python
py_modules=["ml_model_abc"],
```

The code above can be found [here](https://github.com/schmidtbri/ml-model-abc-improvements/blob/master/setup.py#L19).

Next, we take care of the model parameters. The ML model requires that
the model parameters be available for loading at prediction time, the
setup.py file can handle this by adding this line of code:

```python
package_data={'iris_model': ['model_files/svc_model.pickle']},
include_package_data=True,
```

The code above can be found
[here.](https://github.com/schmidtbri/ml-model-abc-improvements/blob/master/setup.py#L25-L28)

This ensures that when the package is installed into an environment, the
model parameters will be copied along with the model\_files folder.

Next, we have to register the iris\_train.py script as an entry point.
This makes it possible to run the training script from the command line
inside of an environment where the iris\_model package is installed:

```python
entry_points={ 'console_scripts': ['iris_train=iris_model.iris_train:main',]
```

The code above can be found [here](https://github.com/schmidtbri/ml-model-abc-improvements/blob/master/setup.py#L29-L32).

Once we have all of this in the setup.py file, we can try to do a pip
install on a new virtual environment. We will install the package
directly from the git repository to keep things simple. The shell
commands to do this are these:

```bash
mkdir example
cd example

# creating a virtual environment
python3 -m venv venv

#activating the virtual environment, on a mac computer
source venv/bin/activate

# installing the iris_model package from the github repository
pip install git+https://github.com/schmidtbri/ml-model-abc-improvements#egg=iris_model
```

Now we can test the installation by starting an interactive Python
interpreter and executing this Python code:

```python
>>> from iris\_model.iris\_predict import IrisModel
>>> model = IrisModel()
>>> model
<iris_model.iris_predict.IrisModel object at 0x105d1e940>
>>> model.input_schema
Schema({'sepal_length': <class 'float'>, 
        'sepal_width': <class 'float'>, 
        'petal_length': <class 'float'>, 
        'petal_width': <class 'float'>})
>>> model.output_schema
Schema({'species': <class 'str'>})
```

Next, we can test the CLI interface for the training code by executing
the command line in the command line:

```bash
iris_train -c=10.0 -gamma=0.01
```

This section showed how to install the iris\_model Python package using
common Python packaging tools, and how to use and retrain the model in
different Python environment.

## Model Metadata in the MLModel Base Class

In the [previous blog
post](https://towardsdatascience.com/a-simple-ml-model-base-class-ab40e2febf13)
we showed an MLModel base class with two required abstract properties:
"input\_schema" and "output\_schema". These two properties were required
to be provided by any class that derived from the MLModel base class and
were used to publish schema metadata about the input and output data of
the model. In order to keep things simple, I chose not to expose more
metadata through class properties, however there are several other
pieces of metadata that would be useful to expose to the outside world.
For example:

-   display_name, a property that returns a display name for the model
-   qualified_name, a property that returns the qualified name of the model, a qualified name is an unambiguous identifier for the model
-   description, a property that returns a description of the model
-   major_version, this property returns the model's major version as a string
-   minor_version, this property returns the model's minor version as a string

These properties are exposed as object properties and can be accessed
the same way as the input\_schema and output\_schema properties. The new
code for the MLModel base class now looks like this:

```python
class MLModel(ABC):
    @property
    @abstractmethod
    def display_name(self):
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def qualified_name(self):
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def description(self):
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def major_version(self):
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def minor_version(self):
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def input_schema(self):
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def output_schema(self):
        raise NotImplementedError()
    
    @abstractmethod
    def __init__(self):
        raise NotImplementedError()
    
    @abstractmethod
    def predict(self, data):
        self.input_schema.validate(data)
```

The code above can be found [here](https://github.com/schmidtbri/ml-model-abc-improvements/blob/master/ml_model_abc.py#L4-L74).

The new MLModel base class looks exactly like the previous
implementation, but now also requires the properties described above to
be published as instance properties.

This metadata is added in the \_\_init\_\_.py file of the iris\_model
package, since it is applicable to the whole package:

```python
# a display name for the model
__display_name__ = "Iris Model"

# returning the package name as the qualified name for the model
__qualified_name__ = __name__.split(".")[0]

# a description of the model
__description__ = "A machine learning model for predicting the species of a flower based on its measurements."
```

The code above can be found
[here](https://github.com/schmidtbri/ml-model-abc-improvements/blob/master/iris_model/__init__.py#L4-L11).

In order to show how a class that derives from the MLModel base class
can publish these properties, we can modify the Iris model example used
in the [previous blog
post](https://towardsdatascience.com/a-simple-ml-model-base-class-ab40e2febf13).
The Iris model class now looks like this:

```python
from ml_model_abc import MLModel
from iris_model import __version_info__, __display_name__, __qualified_name__, __description__

class IrisModel(MLModel):
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
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file = open(os.path.join(dir_path,
                "model_files",
                "svc_model.pickle"), 'rb')
        self._svm_model = pickle.load(file)
        file.close()
        
    def predict(self, data):
        super().predict(data=data)
        X = array([data["sepal_length"],
                   data["sepal_width"],
                   data["petal_length"],
                   data["petal_width"]]).reshape(1, -1)
                    
        y_hat = int(self._svm_model.predict(X)[0])
        targets = ['setosa', 'versicolor', 'virginica']
        species = targets[y_hat]
        return {"species": species}
```

The code above can be found
[here](https://github.com/schmidtbri/ml-model-abc-improvements/blob/master/iris_model/iris_predict.py#L1-L65).

The display name, qualified name, and description properties are set as
string class properties in the IrisModel class, and they are accessed
from the \_\_init\_\_ module. The major and minor version properties are
extracted from the \_\_version\_info\_\_ property.

There can be some situations in which a single Python package will hold
more than one MLModel derived class. In that case the display name,
qualified name, and description metadata would be set individually
within the MLModel derived class itself instead of accessing it from the
package-wide metadata stored in the \_\_init\_\_ module.

The class properties are now easily accessible from the model object, to
show this we can instantiate the object and access the properties:


```python
>>> from iris_model.iris_predict import IrisModel
>>> iris_model = IrisModel()
>>> iris_model.qualified_name
'iris\_model'
>>> iris_model.display_name
'Iris Model'
```

These new metadata properties can now be used to introspect information
about the model more easily, this also makes it possible to more easily
manage many MLModel model objects in the same python process.

## Future Improvements

In this blog post we showed how to do versioning of an ML model using
standard conventions of python packages, however the model parameters of
the Iris model also need to be versioned over time and metadata about
them also needs to be kept. This is a problem that I will tackle in a
future blog post.

Another problem that we did not tackle in this blog post is how to have
a more complex API for ML models. For example, the Iris model is only
allowed to have one predict() method, this makes it impossible to do
more complex operations with the Iris model. In a future blog post I
will show how to modify the ML model base class to allow this.
