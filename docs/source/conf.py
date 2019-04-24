import os
import sys
import json

sys.path.insert(0, os.path.abspath('../../'))


# -- Project information -----------------------------------------------------
project = 'mlmodel-base-class-improvements'
copyright = '2019, Brian Schmidt'
author = 'Brian Schmidt'

# The full version, including alpha/beta/rc tags
release = '0.1.0'


# -- General configuration ---------------------------------------------------
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.intersphinx', 'sphinx-jsonschema', 'sphinxarg.ext']

templates_path = ['_templates']

exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'alabaster'

html_static_path = ['_static']

# this code exports the input and output schema info to a json schema file
# the json schema files are then used to auto generate documentation
from iris_model.iris_predict import IrisSVCModel
input_json_schema_string = json.dumps(IrisSVCModel.input_schema.json_schema("https://example.com/input-schema.json"))
output_json_schema_string = json.dumps(IrisSVCModel.output_schema.json_schema("https://example.com/output-schema.json"))

with open(os.path.abspath('../build/input_schema.json'), "w") as file:
    file.write(input_json_schema_string)

with open(os.path.abspath('../build/output_schema.json'), "w") as file:
    file.write(output_json_schema_string)
