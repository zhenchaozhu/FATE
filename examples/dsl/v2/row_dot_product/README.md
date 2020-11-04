## Row Dot Product Configuration Usage Guide.

This section introduces the dsl and conf for usage of different tasks.

1. Task row dot product:

    dsl: test_row_dot_product_job_dsl.json

    runtime_config : test_row_dot_product_job_conf.json

Users can use following commands to run the task.

    flow job submit -c ${runtime_config} -d ${dsl}

After having finished a successful training task, you can use FATE Board to check output. 