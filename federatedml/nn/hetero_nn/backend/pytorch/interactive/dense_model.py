#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

import tempfile

import numpy as np
import torch
import torch.nn as nn

from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()


class InternalDenseModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(InternalDenseModel, self).__init__()
        LOGGER.debug(f"[DEBUG] InternalDenseModel with shape [{input_dim}, {output_dim}]")

        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=output_dim),
        )

    def _set_parameters(self, weight, bias):
        def init_weights(m):
            if type(m) == nn.Linear:
                with torch.no_grad():
                    m.weight.copy_(torch.tensor(weight))
                    # m.bias.data.fill_(torch.tensor(bias))
                    m.bias.copy_(torch.tensor(bias))

        self.classifier.apply(init_weights)

    def set_parameters(self, parameters):
        weight = parameters["weight"]
        bias = parameters["bias"]
        self._set_parameters(weight, bias)

    def export_model(self):
        f = tempfile.TemporaryFile()
        try:
            torch.save(self.state_dict(), f)
            f.seek(0)
            model_bytes = f.read()
            return model_bytes
        finally:
            f.close()

    def restore_model(self, model_bytes):
        f = tempfile.TemporaryFile()
        f.write(model_bytes)
        f.seek(0)
        self.load_state_dict(torch.load(f))
        f.close()


class DenseModel(object):
    def __init__(self):
        self.input = None
        self.model_weight = None
        # self.model_shape = None
        self.bias = None
        self.model = None
        self.lr = 1.0
        self.layer_config = None
        self.role = "host"
        self.activation_gradient_func = None
        self.activation_func = None
        self.is_empty_model = False
        self.activation_input = None
        self.model_builder = None

    def forward_dense(self, x):
        pass

    def apply_update(self, delta):
        pass

    def get_weight_gradient(self, delta):
        pass

    def build(self, input_shape=None, internal_model_builder=None, restore_stage=False):
        LOGGER.debug(f"[DEBUG] build dense layer with input shape:{input_shape}")
        if not input_shape:
            if self.role == "host":
                raise ValueError("host input is empty!")
            else:
                self.is_empty_model = True
                return

        self.model = internal_model_builder(input_dim=input_shape, output_dim=1)

        if not restore_stage:
            self._init_model_weight(self.model)

    def export_model(self):
        if self.is_empty_model:
            return ''.encode()

        param = {"weight": self.model_weight.T}
        if self.bias is not None:
            param["bias"] = self.bias

        self.model.set_parameters(param)
        return self.model.export_model()

    def restore_model(self, model_bytes):
        if self.is_empty_model:
            return

        LOGGER.debug("model_bytes is {}".format(model_bytes))
        self.model.restore_model(model_bytes)
        self._init_model_weight(self.model)

    def _init_model_weight(self, model):
        LOGGER.debug("[DEBUG] DenseMode._init_model_weight")
        model_params = [param.tolist() for param in model.parameters()]
        self.model_weight = np.array(model_params[0]).T
        self.bias = np.array(model_params[1])

        LOGGER.debug(f"[DEBUG] weight: {self.model_weight}, {self.model_weight.shape}")
        LOGGER.debug(f"[DEBUG] bias: {self.bias}, {self.bias.shape}")

    def forward_activation(self, input_data):
        LOGGER.debug("[DEBUG] DenseModel.forward_activation")
        self.activation_input = input_data
        return input_data

    def backward_activation(self):
        LOGGER.debug("[DEBUG] DenseModel.backward_activation")
        return [1.0]

    def get_weight(self):
        return self.model_weight

    def get_bias(self):
        return self.bias

    def set_learning_rate(self, lr):
        self.lr = lr

    @property
    def empty(self):
        return self.is_empty_model

    @property
    def output_shape(self):
        return self.model_weight.shape[1:]


class GuestDenseModel(DenseModel):

    def __init__(self):
        super(GuestDenseModel, self).__init__()
        self.role = "guest"

    def forward_dense(self, x):
        if self.empty:
            return None

        self.input = x

        output = np.matmul(x, self.model_weight)

        return output

    def get_input_gradient(self, delta):
        if self.empty:
            return None

        error = np.matmul(delta, self.model_weight.T)

        return error

    def get_weight_gradient(self, delta):
        if self.empty:
            return None

        delta_w = np.matmul(delta.T, self.input) / self.input.shape[0]

        return delta_w

    def apply_update(self, delta):
        if self.empty:
            return None

        self.model_weight -= self.lr * delta.T


class PlainHostDenseModel(DenseModel):
    def __init__(self):
        super(PlainHostDenseModel, self).__init__()
        self.role = "host"

    def forward_dense(self, x):
        print("[DEBUG] HostDenseModel.forward_dense")
        """
            x should be encrypted_host_input
        """
        self.input = x
        output = np.matmul(x, self.model_weight)

        if self.bias is not None:
            output += self.bias

        return output

    def get_input_gradient(self, delta, acc_noise=None):
        error = np.matmul(delta, self.model_weight.T)
        return error

    def get_weight_gradient(self, delta):
        delta_w = np.matmul(delta.T, self.input)
        return delta_w

    def update_weight(self, delta):
        self.model_weight -= self.lr * delta.T

    def update_bias(self, delta):
        self.bias -= np.mean(delta, axis=0) * self.lr


class EncryptedHostDenseModel(DenseModel):
    def __init__(self):
        super(EncryptedHostDenseModel, self).__init__()
        self.role = "host"

    def forward_dense(self, x):
        LOGGER.debug("[DEBUG] EncryptedHostDenseModel.forward_dense")
        """
            x should be encrypted host input
        """
        self.input = x

        output = x * self.model_weight

        if self.bias is not None:
            output += self.bias

        return output

    def get_input_gradient(self, delta, acc_noise=None):
        LOGGER.debug("[DEBUG] EncryptedHostDenseModel.get_input_gradient")
        if acc_noise is not None:
            error = delta * (self.model_weight + acc_noise).T
        else:
            error = delta * self.model_weight.T
        return error

    def get_weight_gradient(self, delta):
        LOGGER.debug("[DEBUG] EncryptedHostDenseModel.get_weight_gradient")

        # delta_w = self.input.fast_matmul_2d(delta) / self.input.shape[0]
        delta_w = self.input.fast_matmul_2d(delta)

        return delta_w

    def update_weight(self, delta):
        LOGGER.debug("[DEBUG] EncryptedHostDenseModel.update_weight")
        LOGGER.debug(f"[DEBUG] before weight:{self.model_weight}, delta:{delta}, lr:{self.lr}")
        self.model_weight -= delta * self.lr
        LOGGER.debug(f"[DEBUG] after weight:{self.model_weight}")

    def update_bias(self, delta):
        LOGGER.debug("[DEBUG] EncryptedHostDenseModel.update_bias")
        self.bias -= np.mean(delta, axis=0) * self.lr
