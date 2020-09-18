#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from arch.api.utils import log_utils
from federatedml.nn.hetero_nn.backend.pytorch.interactive.dense_model import EncryptedHostDenseModel
from federatedml.nn.hetero_nn.backend.pytorch.interactive.dense_model import GuestDenseModel, InternalDenseModel
from federatedml.nn.hetero_nn.model.interactive_layer import InterActiveGuestDenseLayer
from federatedml.nn.hetero_nn.model.interactive_layer import InteractiveHostDenseLayer

LOGGER = log_utils.getLogger()


def construct_host_interactive_layer(hetero_nn_param, transfer_variable, partition):
    LOGGER.debug("construct_host_interactive_layer")
    interactive_model = InteractiveHostDenseLayer(hetero_nn_param)
    interactive_model.set_transfer_variable(transfer_variable)
    interactive_model.set_partition(partition)
    return interactive_model


def construct_guest_interactive_layer(hetero_nn_param, transfer_variable, partition, interactive_layer_define,
                                      model_builder):
    LOGGER.debug("construct_guest_interactive_layer")
    interactive_model = InterActiveGuestDenseLayer(hetero_nn_param, interactive_layer_define,
                                                   host_dense_model_builder=construct_host_dense_model,
                                                   guest_dense_model_builder=construct_guest_dense_model,
                                                   model_builder=model_builder)
    interactive_model.set_transfer_variable(transfer_variable)
    interactive_model.set_partition(partition)
    return interactive_model


def construct_host_internal_dense_model(input_dim, output_dim=1):
    return InternalDenseModel(input_dim=input_dim, output_dim=output_dim)


def construct_host_dense_model(host_input_shape, layer_config, model_builder, learning_rate, restore_stage):
    LOGGER.debug("construct_host_dense_model")
    host_model = EncryptedHostDenseModel()
    host_model.build(input_shape=host_input_shape,
                     internal_model_builder=construct_host_internal_dense_model,
                     restore_stage=restore_stage)
    host_model.set_learning_rate(learning_rate)
    return host_model


def construct_guest_dense_model(guest_input_shape, layer_config, model_builder, learning_rate, restore_stage):
    LOGGER.debug("construct_guest_dense_model")
    guest_model = GuestDenseModel()
    guest_model.build(input_shape=guest_input_shape,
                      internal_model_builder=construct_host_internal_dense_model,
                      restore_stage=restore_stage)
    guest_model.set_learning_rate(learning_rate)
    return guest_model
