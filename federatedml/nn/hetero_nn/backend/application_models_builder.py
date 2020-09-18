from collections import OrderedDict

import torch
import torch.nn as nn

from arch.api.utils import log_utils
from federatedml.nn.hetero_nn.backend.host_dann_model import RegionalModel, HostDannModel
from federatedml.nn.hetero_nn.backend.models import RegionalDiscriminator, RegionalAggregator, \
    RegionalFeatureExtractor, TopModel
from federatedml.nn.hetero_nn.model.hetero_nn_bottom_model import HeteroNNBottomModel

LOGGER = log_utils.getLogger()


# TODO: following are for specific application

def construct_guest_top_model(input_shape, optimizer, top_nn_define, loss, metrics, model_builder,
                              data_converter):
    LOGGER.debug("[DEBUG] construct_guest_top_model")
    top_model = TopModel()
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


def construct_host_bottom_model(input_shape, optimizer, layer_config, model_builder, data_converter):
    LOGGER.debug("[DEBUG] construct_host_bottom_model")
    bottom_model = create_host_dann_model(optimizer)
    return bottom_model


# following code is for wiring host bottom DANN models

def create_embedding(size):
    return nn.Embedding(*size, _weight=torch.zeros(*size).normal_(0, 0.01))


def create_embeddings(embedding_meta_dict):
    embedding_dict = dict()
    for key, value in embedding_meta_dict.items():
        embedding_dict[key] = create_embedding(value)
    return embedding_dict


def partition_data(data):
    wide_feat = [data[:, 0].reshape(-1, 1),
                 data[:, 1].reshape(-1, 1),
                 data[:, 2].reshape(-1, 1),
                 data[:, 3].reshape(-1, 1)]
    demo_feat = {"embeddings": OrderedDict({"age_bucket": data[:, 4],
                                            "marital_status": data[:, 5],
                                            "gender": data[:, 6],
                                            "native_country": data[:, 7],
                                            "race": data[:, 8]})}
    emp_feat = {"embeddings": OrderedDict({"age_bucket": data[:, 4],
                                           "workclass": data[:, 9],
                                           "occupation": data[:, 10],
                                           "education": data[:, 11]})}
    demo_emp_feat = {"embeddings": OrderedDict({"age_bucket": data[:, 4],
                                                "gender": data[:, 6],
                                                "occupation": data[:, 10],
                                                "education": data[:, 11]})}
    deep_partition = [demo_feat, emp_feat, demo_emp_feat]
    return wide_feat, deep_partition


def create_region_model_wrapper(extractor_input_dim):
    region_extractor = RegionalFeatureExtractor(input_dims=extractor_input_dim)
    region_aggregator = RegionalAggregator(input_dim=extractor_input_dim[-1])
    region_discriminator = RegionalDiscriminator(input_dim=extractor_input_dim[-1])
    return RegionalModel(extractor=region_extractor,
                         aggregator=region_aggregator,
                         discriminator=region_discriminator)


def create_embedding_dict(embedding_dim=8):
    COLUMNS = {'age_bucket': 11, 'marital_status': 7, 'gender': 2, 'native_country': 43,
               'race': 5, 'workclass': 9, 'occupation': 15, 'education': 17}
    embedding_meta_dict = dict()
    for col, num_values in COLUMNS.items():
        embedding_meta_dict[col] = (num_values, embedding_dim)
    print(f"embedding_meta_dict: \n {embedding_meta_dict}")
    return create_embeddings(embedding_meta_dict)


def create_region_models(input_dims_list):
    wrapper_list = list()
    for input_dim in input_dims_list:
        wrapper_list.append(create_region_model_wrapper(input_dim))
    return wrapper_list


def create_host_dann_model(optimizer_param):
    embedding_dim = 8
    embedding_dict = create_embedding_dict(embedding_dim)
    input_dims_list = [[40, 50, 40, 6],
                       [32, 40, 32, 6],
                       [32, 40, 32, 6]]
    region_models = create_region_models(input_dims_list)

    dann_model = HostDannModel(regional_model_list=region_models, embedding_dict=embedding_dict,
                               partition_data_fn=partition_data,
                               optimizer_param=optimizer_param)
    return dann_model
