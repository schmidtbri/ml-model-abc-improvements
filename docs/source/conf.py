import os
import sys
import json

sys.path.insert(0, os.path.abspath('../../'))

# this code exports the input and output schema info to a json schema file
# the json schema files are then used to auto generate documentation
from iris_model import __version__
from iris_model.iris_predict import IrisModel

input_json_schema_string = json.dumps(IrisModel.input_schema.json_schema("https://example.com/input-schema.json"))
output_json_schema_string = json.dumps(IrisModel.output_schema.json_schema("https://example.com/output-schema.json"))

# create the build directory if it doesn't exist
docs_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if not os.path.exists(os.path.join(docs_path, "build")):
    os.makedirs(os.path.join(docs_path, "build"))

with open(os.path.abspath('../build/input_schema.json'), "w") as file:
    file.write(input_json_schema_string)

with open(os.path.abspath('../build/output_schema.json'), "w") as file:
    file.write(output_json_schema_string)

# -- Project information -----------------------------------------------------
project = 'ML Model Base Class Improvements'
copyright = '2019, Brian Schmidt'
author = 'Brian Schmidt'

# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.intersphinx', 'sphinx-jsonschema', 'sphinxarg.ext']

templates_path = ['_templates']

exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']
