Iris Model Predict Module
=========================

Iris Model Input Schema
***********************

This section describes the input data structure for the predict() method.

.. jsonschema:: ../build/input_schema.json

Iris Model Output Schema
************************

This section describes the output data structure of the predict() method.

.. jsonschema:: ../build/output_schema.json

Iris Model Prediction Code
**************************

.. autoclass:: iris_model.iris_predict.IrisModel
   :members:

   .. automethod:: __init__