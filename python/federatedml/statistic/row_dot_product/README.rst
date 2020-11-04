Row Dot Product
===============

This component does row dot product for features from guest and host.
Please note that Row Dot Product does not have any model output, and thus it should be added to predict pipeline when using it for prediction.

The general procedure is as following:

1. guest encrypts its features and send to host
2. host performs dot product with encrypted guest's features and its own features
3. host sends result dot product back to guest
4. guest decrypts received dot product

How to Use
----------

:params:

    :encrypt_param: EncryptParam Object, note that currently only "paillier" method is supported. Please refer `here <../../param/encrypt_param.py>`_

    :encrypted_mode_calculator_param: EncryptedModeCalculatorParam Object, please refer `here <../../param/encrypted_mode_calculation_param.py>`_

:examples:
    There is an example `conf <../../../../examples/dsl/v2/row_dot_product/test_row_dot_product_job_conf.json>`_
    and `dsl <../../../../examples/dsl/v2/row_dot_product/test_row_dot_product_job_dsl.json>`_

