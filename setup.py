from iris_model import __version__

from setuptools import setup
from os import path

# Get the long description from the README file
with open(path.join(path.abspath(path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='iris_model',
    version=__version__,
    description='A simple ML model example project.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/schmidtbri/ml-model-abc-improvements',
    author='Brian Schmidt',
    author_email='6666331+schmidtbri@users.noreply.github.com',
    py_modules=["ml_model_abc"],
    packages=["iris_model"],
    install_requires=['scikit-learn==0.21.0', 'schema==0.7.0'],
    package_data={'iris_model': ['model_files/svc_model.pickle']},
    include_package_data=True,

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # `pip` to create the appropriate form of executable for the target
    # platform.
    #
    # For example, the following would provide a command called `sample` which
    # executes the function `main` from this package when invoked:
    entry_points={
        'console_scripts': [
            'iris_train=iris_model.iris_train:train',
        ]
    }
)
