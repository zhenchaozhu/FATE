from arch.api.utils import log_utils
from federatedml.nn.hetero_nn.backend.application_models_builder import TopModel
# from federatedml.nn.hetero_nn.model.test.mock_interactive_layer import MockInteractiveHostDenseLayer, \
#     MockInterActiveGuestDenseLayer
from federatedml.nn.hetero_nn.model.hetero_nn_bottom_model import HeteroNNBottomModel
# from federatedml.nn.hetero_nn.backend.hetero_nn_model import HeteroNNKerasGuestModel
# 3from federatedml.nn.hetero_nn.backend.hetero_nn_model import HeteroNNKerasHostModel
# from federatedml.nn.hetero_nn.model.hetero_nn_top_model import HeteroNNTopModel
from federatedml.nn.hetero_nn.model.test.mock_models import MockBottomDenseModel

LOGGER = log_utils.getLogger()


# TODO: following are for specific application

def construct_host_bottom_model(input_shape, optimizer, layer_config, model_builder, data_converter):
    LOGGER.debug("construct mock host bottom model")
    bottom_model = MockBottomDenseModel(input_dim=4, output_dim=3, optimizer_param=optimizer)
    bottom_model.set_data_converter(data_converter)
    return bottom_model


def construct_guest_top_model(input_shape, optimizer, top_nn_define, loss, metrics, model_builder,
                              data_converter):
    LOGGER.debug("construct mock guest top model")
    top_model = TopModel()
    top_model.set_data_converter(data_converter)
    return top_model


def construct_guest_bottom_model(input_shape, optimizer, layer_config, model_builder, data_converter):
    LOGGER.debug("construct mock guest bottom model")
    bottom_model = HeteroNNBottomModel(input_shape=input_shape,
                                       optimizer=optimizer,
                                       layer_config=layer_config,
                                       model_builder=model_builder)
    bottom_model.set_data_converter(data_converter)
    return bottom_model
