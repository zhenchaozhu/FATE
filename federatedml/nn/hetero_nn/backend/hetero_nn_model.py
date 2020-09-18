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

import json

from arch.api.utils import log_utils
from federatedml.nn.hetero_nn.backend.tf_keras.data_generator import KerasSequenceDataConverter
from federatedml.nn.hetero_nn.hetero_nn_model import HeteroNNGuestModel
from federatedml.nn.hetero_nn.hetero_nn_model import HeteroNNHostModel
from federatedml.nn.homo_nn import nn_model
from federatedml.protobuf.generated.hetero_nn_model_meta_pb2 import HeteroNNModelMeta
from federatedml.protobuf.generated.hetero_nn_model_meta_pb2 import OptimizerParam
from federatedml.protobuf.generated.hetero_nn_model_param_pb2 import HeteroNNModelParam

LOGGER = log_utils.getLogger()


class HeteroNNKerasGuestModel(HeteroNNGuestModel):
    def __init__(self, hetero_nn_param, interactive_layer_builder, bottom_model_builder, top_model_builder):
        super(HeteroNNKerasGuestModel, self).__init__()

        self.bottom_model = None
        self.interactive_model = None
        self.top_model = None
        self.bottom_nn_define = None
        self.top_nn_define = None
        self.interactive_layer_define = None
        self.config_type = None
        self.optimizer = None
        self.loss = None
        self.metrics = None
        self.hetero_nn_param = None
        self.transfer_variable = None
        self.model_builder = None
        self.bottom_model_input_shape = 0
        self.top_model_input_shape = None

        self.is_empty = False

        self.set_nn_meta(hetero_nn_param)
        self.model_builder = nn_model.get_nn_builder(config_type=self.config_type)
        self.interactive_layer_builder = interactive_layer_builder
        self.bottom_model_builder = bottom_model_builder
        self.top_model_builder = top_model_builder
        self.data_converter = KerasSequenceDataConverter()

    def set_nn_meta(self, hetero_nn_param):
        self.bottom_nn_define = hetero_nn_param.bottom_nn_define
        self.top_nn_define = hetero_nn_param.top_nn_define
        self.interactive_layer_define = hetero_nn_param.interactive_layer_define
        self.config_type = hetero_nn_param.config_type
        self.optimizer = hetero_nn_param.optimizer
        self.loss = hetero_nn_param.loss
        self.metrics = hetero_nn_param.metrics
        self.hetero_nn_param = hetero_nn_param

    def set_empty(self):
        self.is_empty = True

    def train(self, x, y, epoch, batch_idx, **kwargs):
        LOGGER.debug("Start guest-side training.")

        if not self.is_empty:
            if self.bottom_model is None:
                self.bottom_model_input_shape = x.shape[1]
                self._build_bottom_model()

            guest_bottom_output = self.bottom_model.forward(x, **kwargs)
        else:
            guest_bottom_output = None

        if self.interactive_model is None:
            self._build_interactive_model()

        # get the output of the interactive layer in plain numpy
        LOGGER.debug(f"Guest sends guest_bottom_output:{guest_bottom_output} to interactive layer")
        interactive_output = self.interactive_model.forward(guest_bottom_output, epoch, batch_idx)

        if self.top_model is None:
            self.top_model_input_shape = int(interactive_output.shape[1])
            self._build_top_model()

        # TODO: top model only need to take the gradient of the loss
        LOGGER.debug(f" Guest sends interactive_output:{interactive_output} to top layer")
        gradients = self.top_model.train_and_get_backward_gradient(interactive_output, y)

        LOGGER.debug(f"Guest receives gradients:{gradients.shape} from top layer")
        # LOGGER.debug(f"Guest receives gradients from top layer")
        guest_backward = self.interactive_model.backward(gradients, epoch, batch_idx)
        # LOGGER.debug(f"Guest receives guest_backward:{guest_backward.shape} from interactive layer")
        # LOGGER.debug(f"Guest receives gradients from interactive layer")

        if not self.is_empty:
            self.bottom_model.backward(x, guest_backward, **kwargs)

    def predict(self, x):
        LOGGER.debug("Start guest-side predicting.")
        output = self._predict(x, epoch=0, batch=0)
        predict_prob = self.top_model.predict(output)
        return predict_prob

    def evaluate(self, x, y, epoch, batch):
        LOGGER.debug(f"Start guest-side evaluating at epoch:{epoch} and batch_idx:{batch}.")
        output = self._predict(x, epoch=epoch, batch=batch)
        metrics = self.top_model.evaluate(output, y)
        return metrics

    def _predict(self, x, epoch, batch):
        if not self.is_empty:
            guest_bottom_output = self.bottom_model.predict(x)
        else:
            guest_bottom_output = None

        interactive_output = self.interactive_model.forward(guest_bottom_output, epoch, batch)
        return interactive_output

    def get_hetero_nn_model_param(self):
        model_param = HeteroNNModelParam()
        model_param.is_empty = self.is_empty
        if not self.is_empty:
            model_param.bottom_saved_model_bytes = self.bottom_model.export_model()
        model_param.top_saved_model_bytes = self.top_model.export_model()
        model_param.interactive_layer_param.CopyFrom(self.interactive_model.export_model())

        model_param.bottom_model_input_shape = self.bottom_model_input_shape
        model_param.top_model_input_shape = self.top_model_input_shape

        return model_param

    def set_hetero_nn_model_param(self, model_param):
        self.is_empty = model_param.is_empty
        self.top_model_input_shape = model_param.top_model_input_shape
        self.bottom_model_input_shape = model_param.bottom_model_input_shape
        if not self.is_empty:
            self._restore_bottom_model(model_param.bottom_saved_model_bytes)

        self._restore_interactive_model(model_param.interactive_layer_param)

        self._restore_top_model(model_param.top_saved_model_bytes)

    def get_hetero_nn_model_meta(self):
        model_meta = HeteroNNModelMeta()
        model_meta.config_type = self.config_type

        if self.config_type == "nn":
            for layer in self.bottom_nn_define:
                model_meta.bottom_nn_define.append(json.dumps(layer))

            for layer in self.top_nn_define:
                model_meta.top_nn_define.append(json.dumps(layer))
        elif self.config_type == "keras":
            model_meta.bottom_nn_define.append(json.dumps(self.bottom_nn_define))
            model_meta.top_nn_define.append(json.dumps(self.top_nn_define))

        model_meta.interactive_layer_define = json.dumps(self.interactive_layer_define)
        model_meta.interactive_layer_lr = self.hetero_nn_param.interactive_layer_lr

        model_meta.loss = self.loss

        """
        for metric in self.metrics:
            model_meta.metrics.append(metric)
        """

        optimizer_param = OptimizerParam()
        optimizer_param.optimizer = self.optimizer.optimizer
        optimizer_param.kwargs = json.dumps(self.optimizer.kwargs)

        model_meta.optimizer_param.CopyFrom(optimizer_param)

        return model_meta

    def set_hetero_nn_model_meta(self, model_meta):
        self.config_type = model_meta.config_type

        if self.config_type == "nn":
            self.bottom_nn_define = []
            self.top_nn_define = []

            for layer in model_meta.bottom_nn_define:
                self.bottom_nn_define.append(json.loads(layer))

            for layer in model_meta.top_nn_define:
                self.top_nn_define.append(json.loads(layer))
        elif self.config_type == 'keras':
            self.bottom_nn_define = json.loads(model_meta.bottom_nn_define[0])
            self.top_nn_define = json.loads(model_meta.top_nn_define[0])

        self.interactive_layer_define = json.loads(model_meta.interactive_layer_define)
        self.loss = model_meta.loss

        """
        self.metrics = []
        for metric in self.metrics:
            self.metrics.append(metric)
        """

        self.optimizer.optimizer = model_meta.optimizer_param.optimizer
        self.optimizer.kwargs = json.loads(model_meta.optimizer_param.kwargs)

    def set_transfer_variable(self, transfer_variable):
        self.transfer_variable = transfer_variable

    def set_partition(self, partition=1):
        self.partition = partition
        if self.interactive_model is not None:
            self.interactive_model.set_partition(self.partition)

    def _build_bottom_model(self):
        LOGGER.debug("[DEBUG] HeteroNNKerasGuestModel._build_bottom_model")
        self.bottom_model = self.bottom_model_builder(input_shape=self.bottom_model_input_shape,
                                                      optimizer=self.optimizer,
                                                      layer_config=self.bottom_nn_define,
                                                      model_builder=self.model_builder,
                                                      data_converter=self.data_converter)

    def _restore_bottom_model(self, model_bytes):
        self._build_bottom_model()
        self.bottom_model.restore_model(model_bytes)

    def _build_top_model(self):
        LOGGER.debug("[DEBUG] HeteroNNKerasGuestModel._build_top_model")
        self.top_model = self.top_model_builder(input_shape=self.top_model_input_shape,
                                                optimizer=self.optimizer,
                                                top_nn_define=self.top_nn_define,
                                                loss=self.loss,
                                                metrics=self.metrics,
                                                model_builder=self.model_builder,
                                                data_converter=self.data_converter)

    def _restore_top_model(self, model_bytes):
        self._build_top_model()
        self.top_model.restore_model(model_bytes)

    def _build_interactive_model(self):
        LOGGER.debug("[DEBUG] HeteroNNKerasGuestModel._build_interactive_model")
        self.interactive_model = self.interactive_layer_builder(hetero_nn_param=self.hetero_nn_param,
                                                                transfer_variable=self.transfer_variable,
                                                                partition=self.partition,
                                                                interactive_layer_define=self.interactive_layer_define,
                                                                model_builder=self.model_builder)

    def _restore_interactive_model(self, interactive_model_param):
        self._build_interactive_model()

        self.interactive_model.restore_model(interactive_model_param)


class HeteroNNKerasHostModel(HeteroNNHostModel):
    def __init__(self, hetero_nn_param, interactive_layer_builder, bottom_model_builder):
        super(HeteroNNKerasHostModel, self).__init__()

        self.bottom_model_input_shape = None
        self.bottom_model = None
        self.interactive_model = None

        self.bottom_nn_define = None
        self.config_type = None
        self.optimizer = None
        self.hetero_nn_param = None

        self.set_nn_meta(hetero_nn_param)

        self.model_builder = nn_model.get_nn_builder(config_type=self.config_type)
        self.interactive_layer_builder = interactive_layer_builder
        self.bottom_model_builder = bottom_model_builder
        self.data_converter = KerasSequenceDataConverter()

        self.transfer_variable = None

        # control whether host performs local training without the participation of guest
        self.is_local_train = False

    def set_nn_meta(self, hetero_nn_param):
        self.bottom_nn_define = hetero_nn_param.bottom_nn_define
        self.config_type = hetero_nn_param.config_type
        self.optimizer = hetero_nn_param.optimizer
        self.hetero_nn_param = hetero_nn_param
        self.is_local_train = True if hetero_nn_param["is_local_train"] else False

    def _build_bottom_model(self):
        LOGGER.debug("[DEBUG] HeteroNNKerasHostModel.build_bottom_model")
        self.bottom_model = self.bottom_model_builder(input_shape=self.bottom_model_input_shape,
                                                      optimizer=self.optimizer,
                                                      layer_config=self.bottom_nn_define,
                                                      model_builder=self.model_builder,
                                                      data_converter=self.data_converter)

    def _restore_bottom_model(self, model_bytes):
        self._build_bottom_model()
        self.bottom_model.restore_model(model_bytes)

    def _build_interactive_model(self):
        LOGGER.debug("[DEBUG] HeteroNNKerasHostModel._build_interactive_model")
        self.interactive_model = self.interactive_layer_builder(hetero_nn_param=self.hetero_nn_param,
                                                                transfer_variable=self.transfer_variable,
                                                                partition=self.partition)

    def _restore_interactive_model(self, interactive_layer_param):
        self._build_interactive_model()
        self.interactive_model.restore_model(interactive_layer_param)

    def set_transfer_variable(self, transfer_variable):
        self.transfer_variable = transfer_variable

    def set_partition(self, partition=1):
        self.partition = partition
        if self.interactive_model is not None:
            self.interactive_model.set_partition(self.partition)

        LOGGER.debug("set_partition, partition num is {}".format(self.partition))

    def get_hetero_nn_model_meta(self):
        model_meta = HeteroNNModelMeta()

        model_meta.config_type = self.config_type

        if self.config_type == "nn":
            for layer in self.bottom_nn_define:
                model_meta.bottom_nn_define.append(json.dumps(layer))

        elif self.config_type == "keras":
            model_meta.bottom_nn_define.append(json.dumps(self.bottom_nn_define))

        model_meta.interactive_layer_lr = self.hetero_nn_param.interactive_layer_lr

        optimizer_param = OptimizerParam()
        optimizer_param.optimizer = self.optimizer.optimizer
        optimizer_param.kwargs = json.dumps(self.optimizer.kwargs)

        model_meta.optimizer_param.CopyFrom(optimizer_param)

        return model_meta

    def set_hetero_nn_model_meta(self, model_meta):
        self.config_type = model_meta.config_type

        if self.config_type == "nn":
            self.bottom_nn_define = []

            for layer in model_meta.bottom_nn_define:
                self.bottom_nn_define.append(json.loads(layer))

        elif self.config_type == 'keras':
            self.bottom_nn_define = json.loads(model_meta.bottom_nn_define[0])

        self.optimizer.optimizer = model_meta.optimizer_param.optimizer
        self.optimizer.kwargs = json.loads(model_meta.optimizer_param.kwargs)

    def set_hetero_nn_model_param(self, model_param):
        self.bottom_model_input_shape = model_param.bottom_model_input_shape
        self._restore_bottom_model(model_param.bottom_saved_model_bytes)
        self._restore_interactive_model(model_param.interactive_layer_param)

    def get_hetero_nn_model_param(self):
        model_param = HeteroNNModelParam()
        model_param.bottom_saved_model_bytes = self.bottom_model.export_model()
        model_param.interactive_layer_param.CopyFrom(self.interactive_model.export_model())
        return model_param

    def train(self, x, y, epoch, batch_idx, **kwargs):
        LOGGER.debug(f"host train mode:{self.is_local_train}.")

        kwargs["current_epoch"] = epoch
        kwargs["batch_idx"] = batch_idx

        if self.bottom_model is None:
            self.bottom_model_input_shape = x.shape[1]
            self._build_bottom_model()
            if not self.is_local_train:
                self._build_interactive_model()

        if self.is_local_train:
            self._local_train(x, y, **kwargs)
        else:
            self._train(x, epoch, batch_idx, **kwargs)

    def _local_train(self, x, y, **kwargs):
        LOGGER.debug("Start local training.")
        self.bottom_model.local_train(x, y, **kwargs)

    def _train(self, x, epoch, batch_idx, **kwargs):
        LOGGER.debug("Start host-side training of federation.")

        host_bottom_output = self.bottom_model.forward(x, **kwargs)
        LOGGER.debug(f"Host sends host_bottom_output:{host_bottom_output} to interactive layer")
        self.interactive_model.forward(host_bottom_output, epoch, batch_idx)
        host_gradient = self.interactive_model.backward(epoch, batch_idx)
        # host_gradient in plain numpy to be sent to host bottom model
        LOGGER.debug(f"Host receives host_gradient:{host_gradient.shape} from interactive layer")

        if host_gradient is not None:
            LOGGER.debug(f"Host starts back-propagation of host bottom model")
            self.bottom_model.backward(x, host_gradient, **kwargs)

    def evaluate(self, x, y, epoch, batch_idx):
        metrics = None
        if self.is_local_train:
            LOGGER.debug(f"Start host-side local evaluation with epoch:{epoch} and batch_idx:{batch_idx}.")
            metrics = self.bottom_model.local_evaluate(x, y)
        else:
            LOGGER.debug(f"Start host-side federated evaluation with epoch:{epoch} and batch_idx:{batch_idx}.")
            host_bottom_output = self.bottom_model.predict(x)
            self.interactive_model.forward(host_bottom_output, epoch, batch_idx)
        return metrics

    def predict(self, x):
        LOGGER.debug("Start host-side predicting.")
        host_bottom_output = self.bottom_model.predict(x)
        self.interactive_model.forward(host_bottom_output)
