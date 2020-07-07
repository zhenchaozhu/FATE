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

import numpy as np

from arch.api.utils import log_utils
from federatedml.nn.hetero_nn.model.test.mock_models import MockDenseModel

LOGGER = log_utils.getLogger()


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
        # self.activation_placeholder_name = "activation_placeholder" + str(uuid.uuid1())
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

    # def set_sess(self, sess):
    #     self.sess = sess

    def build(self, input_shape=None, layer_config=None, model_builder=None, restore_stage=False):
        print(f"[DEBUG] build dense layer with input shape:{input_shape}")
        if not input_shape:
            if self.role == "host":
                raise ValueError("host input is empty!")
            else:
                self.is_empty_model = True
                return

        # self.model_builder = model_builder
        self.layer_config = layer_config
        self.model = MockDenseModel(input_dim=input_shape, output_dim=1)

        # dense_layer = self.model.get_layer_by_index(0)
        if not restore_stage:
            self._init_model_weight(self.model)

        # if self.role == "host":
        #     self.activation_func = dense_layer.activation
        #     self.__build_activation_layer_gradients_func(dense_layer)

    def export_model(self):
        if self.is_empty_model:
            return ''.encode()

        layer_weights = [self.model_weight]
        if self.bias is not None:
            layer_weights.append(self.bias)

        # TODO: refactor following line
        self.model.set_layer_weights_by_index(0, layer_weights)
        return self.model.export_model()
        # return None

    def restore_model(self, model_bytes):
        if self.is_empty_model:
            return

        LOGGER.debug("model_bytes is {}".format(model_bytes))
        self.model.restore_model(model_bytes)
        self._init_model_weight(self.model, restore_stage=True)

    def _init_model_weight(self, model, restore_stage=False):
        print("[DEBUG] DenseMode._init_model_weight")
        model_params = [param.tolist() for param in model.parameters()]
        self.model_weight = np.array(model_params[0]).T
        # self.model_shape = self.model_weight.shape
        self.bias = np.array(model_params[1])

        print("[DEBUG] weight: ", self.model_weight, self.model_weight.shape)
        print("[DEBUG] bias: ", self.bias, self.bias.shape)

    # def __build_activation_layer_gradients_func(self, dense_layer):
    #     shape = dense_layer.output_shape
    #     dtype = dense_layer.get_weights()[0].dtype
    #
    #     input_data = tf.placeholder(shape=shape,
    #                                 dtype=dtype,
    #                                 name=self.activation_placeholder_name)
    #
    #     self.activation_gradient_func = gradients(dense_layer.activation(input_data), input_data)

    def forward_activation(self, input_data):
        print("[DEBUG] DenseModel.forward_activation")

        self.activation_input = input_data

        # output = self.activation_func(input_data)
        # if not isinstance(output, np.ndarray):
        #     output = self.sess.run(output)

        return input_data

    def backward_activation(self):
        print("[DEBUG] DenseModel.backward_activation")

        # placeholder = tf.get_default_graph().get_tensor_by_name(":".join([self.activation_placeholder_name, "0"]))
        # return self.sess.run(self.activation_gradient_func,
        #                      feed_dict={placeholder: self.activation_input})
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
        print("x:", x, x.shape)
        print("model_weight:", self.model_weight, self.model_weight.shape)
        output = np.matmul(x, self.model_weight)
        # output = x * self.model_weight
        print("output:", output, output.shape)

        if self.bias is not None:
            output += self.bias

        print("output with bias:", output, output.shape)
        return output

    def get_input_gradient(self, delta, acc_noise=None):
        print("[DEBUG] HostDenseModel.get_input_gradient")
        print("model_weight:", self.model_weight, self.model_weight.shape)
        print("delta:", delta, delta.shape)
        # error = delta * (self.model_weight + acc_noise).T
        error = np.matmul(delta, self.model_weight.T)
        print("error:", error, error.shape)
        return error

    def get_weight_gradient(self, delta):
        print("[DEBUG] HostDenseModel.get_weight_gradient")
        print("input:", self.input, self.input.shape)
        print("delta:", delta, delta.shape)
        # delta_w = self.input.fast_matmul_2d(delta) / self.input.shape[0]
        # delta_w = self.input * delta
        delta_w = np.matmul(delta.T, self.input)
        # delta_w = np.matmul(delta.T, self.input) / self.input.shape[0]

        print("delta_w:", delta_w, delta_w.shape)
        # delta_w = np.sum(delta_w, axis=0)
        # print("delta_w:", delta_w)

        return delta_w

    def update_weight(self, delta):
        print("[DEBUG] HostDenseModel.update_weight")
        print("delta:", delta, delta.shape)
        # self.model_weight -= delta * self.lr
        self.model_weight -= self.lr * delta.T
        print("*[DEBUG] updated weight:", self.model_weight)

    def update_bias(self, delta):
        print("[DEBUG] HostDenseModel.update_bias")
        print("delta:", delta, delta.shape)
        self.bias -= np.mean(delta, axis=0) * self.lr
        print("*[DEBUG] updated bias:", self.bias)


class EncryptedHostDenseModel(DenseModel):
    def __init__(self):
        super(EncryptedHostDenseModel, self).__init__()
        self.role = "host"

    def forward_dense(self, x):
        print("[DEBUG] EncryptedHostDenseModel.forward_dense")
        """
            x should be encrypted_host_input
        """
        self.input = x

        output = x * self.model_weight

        if self.bias is not None:
            output += self.bias

        return output

    def get_input_gradient(self, delta, acc_noise=None):
        print("[DEBUG] EncryptedHostDenseModel.get_input_gradient")
        error = delta * (self.model_weight + acc_noise).T
        return error

    def get_weight_gradient(self, delta):
        print("[DEBUG] EncryptedHostDenseModel.get_weight_gradient")

        delta_w = self.input.fast_matmul_2d(delta) / self.input.shape[0]

        return delta_w

    def update_weight(self, delta):
        print("[DEBUG] EncryptedHostDenseModel.update_weight")
        self.model_weight -= delta * self.lr

    def update_bias(self, delta):
        print("[DEBUG] EncryptedHostDenseModel.update_bias")
        self.bias -= np.mean(delta, axis=0) * self.lr
