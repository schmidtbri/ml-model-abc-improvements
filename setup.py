from distutils.core import setup

from iris_model import __version__

setup(
    name='iris_model',
    version=__version__,
    packages=['iris_model'],
    license='MIT',
    long_description=open('README.txt').read(),
)
