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
from federatedml.nn.hetero_nn.backend.application_models_builder import construct_guest_top_model, \
    construct_host_bottom_model, construct_guest_bottom_model
# from federatedml.nn.hetero_nn.backend.mock_models_builder import construct_guest_top_model, \
#     construct_host_bottom_model, construct_guest_bottom_model
from federatedml.nn.hetero_nn.backend.hetero_nn_model import HeteroNNKerasGuestModel
from federatedml.nn.hetero_nn.backend.hetero_nn_model import HeteroNNKerasHostModel
from federatedml.nn.hetero_nn.backend.interactive_layer_builder import construct_host_interactive_layer, \
    construct_guest_interactive_layer

LOGGER = log_utils.getLogger()


def model_builder(role="guest", hetero_nn_param=None, backend="keras"):
    if backend != "keras":
        raise ValueError("Only support keras backend in this version!")

    if role == "guest":
        return HeteroNNKerasGuestModel(hetero_nn_param,
                                       interactive_layer_builder=construct_guest_interactive_layer,
                                       top_model_builder=construct_guest_top_model,
                                       bottom_model_builder=construct_guest_bottom_model)
    elif role == "host":
        return HeteroNNKerasHostModel(hetero_nn_param,
                                      interactive_layer_builder=construct_host_interactive_layer,
                                      bottom_model_builder=construct_host_bottom_model)
