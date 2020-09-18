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
from torch.utils.data import DataLoader

from arch.api.utils import log_utils
from fate_flow.entity.metric import Metric
from federatedml.framework.hetero.procedure import batch_generator
from federatedml.nn.hetero_nn.backend.model_builder import model_builder
from federatedml.nn.hetero_nn.dataset import SimpleDataset
from federatedml.nn.hetero_nn.hetero_nn_base import HeteroNNBase
from federatedml.nn.hetero_nn.regularization import EarlyStoppingCheckPoint
from federatedml.protobuf.generated.hetero_nn_model_meta_pb2 import HeteroNNMeta
from federatedml.protobuf.generated.hetero_nn_model_param_pb2 import HeteroNNParam

LOGGER = log_utils.getLogger()
MODELMETA = "HeteroNNHostMeta"
MODELPARAM = "HeteroNNHostParam"


class HeteroNNHost(HeteroNNBase):
    def __init__(self):
        super(HeteroNNHost, self).__init__()

        self.batch_generator = batch_generator.Host()
        self.model = None

        self.input_shape = None
        self.validation_strategy = None
        self.val_data_x = []
        self.val_data_y = []
        self.history_loss = None

    def _init_model(self, hetero_nn_param):
        super(HeteroNNHost, self)._init_model(hetero_nn_param)
        self.is_local_train = True if hetero_nn_param["is_local_train"] else False
        self.inference_for_train = True if hetero_nn_param["inference_for_train"] else False

    # TODO: Question: is it possible that this "export_model" function will not be called by fate_flow ?
    #  is it will not be called in the predict mode by fate_flow ?
    def export_model(self):
        if self.model is None:
            return

        return {MODELMETA: self._get_model_meta(),
                MODELPARAM: self._get_model_param()}

    # TODO: where/when is this load_model called? before the predict is called ?
    def load_model(self, model_dict):
        model_dict = list(model_dict["model"].values())[0]
        param = model_dict.get(MODELPARAM)
        meta = model_dict.get(MODELMETA)

        self._build_model()
        self._restore_model_meta(meta)
        self._restore_model_param(param)

    def _build_model(self):
        self.model = model_builder("host", self.hetero_nn_param)
        self.model.set_transfer_variable(self.transfer_variable)

    def fit(self, train_data, validate_data=None):
        if self.is_local_train:
            # TODO: add doc
            self.local_fit(train_data, validate_data, self.batch_size)
        else:
            # TODO: add doc
            self.federate_fit(train_data, validate_data)

    def local_fit(self, train_data, validate_data, batch_size):
        """
            perform model training with only host's data
        """

        train_data_loader = self.prepare_local_data_loader(data_inst=train_data, batch_size=batch_size)
        valid_data_loader = self.prepare_local_data_loader(data_inst=validate_data, batch_size=batch_size)
        iterations_per_epoch = len(train_data_loader)
        patience = 2 * iterations_per_epoch

        self._set_loss_callback_info()
        early_stopping_checkpoint = EarlyStoppingCheckPoint(learner=self,
                                                            model=self.model,
                                                            monitor_value="auc",
                                                            patience=patience)
        early_stopping_checkpoint.on_train_begin()

        kwargs = dict()
        kwargs["max_epochs"] = 200
        kwargs["num_batch"] = iterations_per_epoch
        cur_epoch = 0
        validation_batch_frequency = 40
        self.history_loss = list()
        while cur_epoch < self.epochs:
            epoch_loss = 0
            # for data_x, data_y in train_data_loader:
            # for batch_idx in range(len(self.data_x)):
            for batch_idx in range(iterations_per_epoch):
                data_x, data_y = next(train_data_loader)
                self.model.train(data_x, data_y, cur_epoch, batch_idx, **kwargs)

                # validate
                if (batch_idx + 1) % validation_batch_frequency == 0:
                    eval_loss, metrics_dict = self.local_validate(valid_data_loader, cur_epoch, batch_idx)
                    epoch_loss = eval_loss
                    early_stopping_checkpoint.on_validation_end(cur_epoch, batch_idx, metrics_dict)

                res = early_stopping_checkpoint.on_batch_end()
                if res['stop_training']:
                    break

            res = early_stopping_checkpoint.on_epoch_end()
            if res['stop_training']:
                break

            self.callback_metric("loss", "train", [Metric(cur_epoch, epoch_loss)])
            self.history_loss.append(epoch_loss)
            cur_epoch += 1

        if cur_epoch == self.epochs:
            LOGGER.debug("Training process reach max training epochs {} and not converged".format(self.epochs))

        early_stopping_checkpoint.on_train_end()

    def local_validate(self, valid_data_loader, cur_epoch, batch_idx):
        eval_loss_list = []
        auc_accumulate = 0
        acc_accumulate = 0
        num_samples = 0
        iterations_per_epoch = len(valid_data_loader)
        for batch_idx in range(iterations_per_epoch):
            val_data_x, val_data_y = next(valid_data_loader)
            LOGGER.debug(
                f"data_x shape: {val_data_x.shape}, data_y shape {val_data_y.shape}, epoch:{cur_epoch}, "
                f"batch_idx:{batch_idx}")
            metrics = self.model.evaluate(val_data_x, val_data_y, cur_epoch, batch_idx)
            eval_loss_list.append(metrics["loss"])
            batch_size = len(val_data_y)
            num_samples += batch_size
            auc_accumulate += metrics["auc"] * batch_size
            acc_accumulate += metrics["acc"] * batch_size
        eval_auc = auc_accumulate / num_samples
        eval_acc = acc_accumulate / num_samples
        eval_loss = np.mean(eval_loss_list)
        LOGGER.info(f"epoch:{cur_epoch}, batch_idx:{batch_idx}, eval_loss:{eval_loss}, "
                    f"auc is {eval_auc}, acc is {eval_acc}")
        return eval_loss, {"auc": eval_auc, "acc": eval_acc}

    def federate_fit(self, train_data, validate_data, make_inference_only=False):
        LOGGER.debug(f"start fitting at host.")
        self.validation_strategy = self.init_validation_strategy(train_data, validate_data)
        self._build_model()
        self.prepare_batch_data(self.batch_generator, train_data)
        self.prepare_batch_val_data(self.batch_generator, validate_data)

        kwargs = dict()
        kwargs["max_epochs"] = 200
        kwargs["num_batch"] = len(self.data_x)
        cur_epoch = 0
        validation_batch_frequency = 40
        while cur_epoch < self.epochs:
            for batch_idx in range(len(self.data_x)):
                if make_inference_only:
                    self.model.predict(self.data_x[batch_idx])
                else:
                    self.model.train(self.data_x[batch_idx], None, cur_epoch, batch_idx, **kwargs)

                # validate
                if (batch_idx + 1) % validation_batch_frequency == 0:
                    self.reset_flowid(str(cur_epoch) + "_" + str(batch_idx))
                    for batch_val_idx in range(len(self.val_data_x)):
                        val_data_x = self.val_data_x[batch_val_idx]
                        LOGGER.debug(
                            f"data_x shape: {val_data_x.shape}, epoch:{cur_epoch}, batch_val_idx:{batch_val_idx}")
                        self.model.evaluate(val_data_x, None, cur_epoch, batch_val_idx)
                    self.recovery_flowid()

            if self.validation_strategy:
                self.validation_strategy.validate(self, cur_epoch)
                if self.validation_strategy.need_stop():
                    LOGGER.debug('early stopping triggered')
                    break

            is_converge = self.transfer_variable.is_converge.get(idx=0, suffix=(cur_epoch,))

            if is_converge:
                LOGGER.debug("Training process is converged in epoch {}".format(cur_epoch))
                break

            cur_epoch += 1

        # when we are in make_inference_only, we do not need to save model
        if (not make_inference_only) and self.validation_strategy and self.validation_strategy.has_saved_best_model():
            self.load_model(self.validation_strategy.cur_best_model)

    def predict(self, data_inst):
        # TODO: since currently FATE does not support fine-tune, this is a temporary implementation for
        #  scenario where we only want to train the model of guest while freeze the model of host.
        if len(data_inst) == 2:
            # TODO: make data_inst a list of data sources by configuration in dsl file
            predict_data = data_inst[0]
            validate_data = data_inst[1]
            # TODO: question: automatically load/restore model before prediction?
            self.validation_strategy = self.init_validation_strategy(predict_data, validate_data)
            self.prepare_batch_data(self.batch_generator, predict_data)
            self.prepare_batch_val_data(self.batch_generator, validate_data)
            self.federate_fit(make_inference_only=True)
        else:
            test_x = self._load_data(data_inst)
            self.set_partition(data_inst)
            self.model.predict(test_x)

    def prepare_local_data_loader(self, data_inst, batch_size):
        x, y = self._load_data(data_inst)
        return DataLoader(dataset=SimpleDataset(x, y), batch_size=batch_size, shuffle=True)

    def prepare_batch_data(self, batch_generator, data_inst):
        batch_generator.initialize_batch_generator(data_inst)
        batch_data_generator = batch_generator.generate_batch_data()

        for batch_data in batch_data_generator:
            batch_x, batch_y = self._load_data(batch_data)
            self.data_x.append(batch_x)
            self.data_y.append(batch_y)

        self.set_partition(data_inst)

    def prepare_batch_val_data(self, batch_generator, data_inst):
        batch_generator.initialize_batch_generator(data_inst, suffix=("val",))
        batch_data_generator = batch_generator.generate_batch_data()

        for batch_data in batch_data_generator:
            batch_x, batch_y = self._load_data(batch_data)
            self.val_data_x.append(batch_x)
            self.val_data_y.append(batch_y)

        # self.set_partition(data_inst)

    def _load_data(self, data_inst):
        data = list(data_inst.collect())
        data_keys = [key for (key, val) in data]
        data_keys_map = dict(zip(sorted(data_keys), range(len(data_keys))))
        x = [None for _ in range(len(data_keys))]
        y = [None for _ in range(len(data_keys))]

        for key, inst in data:
            idx = data_keys_map[key]
            x[idx] = inst.features
            y[idx] = inst.label

            if self.input_shape is None:
                self.input_shape = inst.features.shape

        x = np.asarray(x)
        y = np.asarray(y)

        return x, y

    def _get_model_meta(self):
        model_meta = HeteroNNMeta()
        model_meta.batch_size = self.batch_size
        model_meta.hetero_nn_model_meta.CopyFrom(self.model.get_hetero_nn_model_meta())

        return model_meta

    def _get_model_param(self):
        model_param = HeteroNNParam()
        model_param.hetero_nn_model_param.CopyFrom(self.model.get_hetero_nn_model_param())
        model_param.best_iteration = -1 if self.validation_strategy is None else self.validation_strategy.best_iteration

        # model_param.num_label = self.num_label
        for loss in self.history_loss:
            model_param.history_loss.append(loss)

        return model_param
