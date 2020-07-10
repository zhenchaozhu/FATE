import torch
import torch.nn as nn

from arch.api.utils import log_utils
# from federatedml.nn.hetero_nn.model.test.mock_interactive_layer import MockInteractiveHostDenseLayer, \
#     MockInterActiveGuestDenseLayer
from federatedml.nn.hetero_nn.model.hetero_nn_bottom_model import HeteroNNBottomModel
# from federatedml.nn.hetero_nn.backend.hetero_nn_model import HeteroNNKerasGuestModel
# 3from federatedml.nn.hetero_nn.backend.hetero_nn_model import HeteroNNKerasHostModel
from federatedml.nn.hetero_nn.model.interactive_layer import InterActiveGuestDenseLayer
from federatedml.nn.hetero_nn.model.interactive_layer import InteractiveHostDenseLayer
# from federatedml.nn.hetero_nn.model.hetero_nn_top_model import HeteroNNTopModel
from federatedml.nn.hetero_nn.model.test.mock_models import MockBottomDenseModel, MockTopModel

LOGGER = log_utils.getLogger()


# TODO: following are for specific application

def construct_host_bottom_model(input_shape, optimizer, layer_config, model_builder, data_converter):
    LOGGER.debug("[DEBUG] construct_host_bottom_model")
    # bottom_model = HeteroNNBottomModel(input_shape=bottom_model_input_shape,
    #                                         optimizer=optimizer,
    #                                         layer_config=bottom_nn_define,
    #                                         model_builder=model_builder)

    bottom_model = MockBottomDenseModel(input_dim=4, output_dim=3, optimizer_param=optimizer)
    bottom_model.set_data_converter(data_converter)
    return bottom_model


def construct_host_interactive_layer(hetero_nn_param, transfer_variable, partition):
    LOGGER.debug("construct_host_interactive_layer")
    interactive_model = InteractiveHostDenseLayer(hetero_nn_param)
    # interactive_model = MockInteractiveHostDenseLayer(hetero_nn_param)
    interactive_model.set_transfer_variable(transfer_variable)
    interactive_model.set_partition(partition)
    return interactive_model


def construct_guest_interactive_layer(hetero_nn_param, transfer_variable, partition, interactive_layer_define,
                                      model_builder):
    LOGGER.debug("construct_guest_interactive_layer")
    interactive_model = InterActiveGuestDenseLayer(hetero_nn_param, interactive_layer_define,
                                                   model_builder=model_builder)
    # interactive_model = MockInterActiveGuestDenseLayer(hetero_nn_param)
    interactive_model.set_transfer_variable(transfer_variable)
    interactive_model.set_partition(partition)
    return interactive_model


def construct_guest_top_model(top_model_input_shape, optimizer, top_nn_define, loss, metrics, model_builder,
                              data_converter):
    LOGGER.debug("[DEBUG] construct_guest_top_model")
    # top_model = HeteroNNTopModel(input_shape=self.top_model_input_shape,
    #                                   optimizer=self.optimizer,
    #                                   layer_config=self.top_nn_define,
    #                                   loss=self.loss,
    #                                   metrics=self.metrics,
    #                                   model_builder=self.model_builder)

    top_model = MockTopModel()

    top_model.set_data_converter(data_converter)
    return top_model


def construct_guest_bottom_model(input_shape, optimizer, layer_config, model_builder, data_converter):
    LOGGER.debug("[DEBUG] construct_guest_bottom_model")
    bottom_model = HeteroNNBottomModel(input_shape=input_shape,
                                       optimizer=optimizer,
                                       layer_config=layer_config,
                                       model_builder=model_builder)
    bottom_model.set_data_converter(data_converter)
    return bottom_model


def create_embedding(size):
    # return nn.Parameter(torch.zeros(*size).normal_(0, 0.01))
    return nn.Embedding(*size, _weight=torch.zeros(*size).normal_(0, 0.01))


def create_embeddings(embedding_meta_dict):
    embedding_dict = dict()
    for key, value in embedding_meta_dict.items():
        embedding_dict[key] = create_embedding(value)
    return embedding_dict


def create_embedding_dict(embedding_dim=8):
    COLUMNS = {'age_bucket': 11, 'marital_status': 7, 'gender': 2, 'native_country': 43,
               'race': 5, 'workclass': 9, 'occupation': 15, 'education': 17}
    embedding_meta_dict = dict()
    for col, num_values in COLUMNS.items():
        embedding_meta_dict[col] = (num_values, embedding_dim)
    print(f"embedding_meta_dict: \n {embedding_meta_dict}")
    return create_embeddings(embedding_meta_dict)


def create_regional_model(extractor_output_dim, extractor_input_dim):
    extractor = CensusRegionFeatureExtractorDense(input_dims=extractor_input_dim)
    region_classifier = CensusRegionClassifier(input_dim=extractor_output_dim)
    discriminator = CensusRegionDiscriminator(input_dim=extractor_output_dim)
    return RegionModelWrapper(extractor=extractor,
                              aggregator=region_classifier,
                              discriminator=discriminator)


def create_region_models(embedding_dim):
    input_dim_list = [5 * embedding_dim, 4 * embedding_dim, 4 * embedding_dim]
    wrapper_list = list()
    for input_dim in input_dim_list:
        wrapper_list.append(create_regional_model(10, input_dim))
    return wrapper_list


def create_global_model():
    embedding_dim = 8
    embedding_dict = create_embedding_dict(embedding_dim)
    region_wrapper_list = create_region_models(embedding_dim)

    global_input_dim = 7
    wrapper = GlobalModelWrapper(classifier, region_wrapper_list, embedding_dict)
    return wrapper
