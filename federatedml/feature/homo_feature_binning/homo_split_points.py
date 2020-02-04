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

from federatedml.transfer_variable.transfer_class.homo_feature_binning_transfer_variable import \
    HomoFeatureBinningTransferVariable
from federatedml.util import consts
from federatedml.param.feature_binning_param import FeatureBinningParam
from federatedml.feature.binning.quantile_binning import QuantileBinning
from arch.api.utils import log_utils
import numpy as np
from federatedml.util import abnormal_detection

LOGGER = log_utils.getLogger()


class HomoSplitPointCalculator(object):
    def __init__(self, role, flowid='homo_split_pointer', bin_method=consts.QUANTILE):
        self.role = role
        self.transfer_variable = HomoFeatureBinningTransferVariable()
        self.transfer_variable.set_flowid(flowid)
        self.bin_method = bin_method

    def set_flowid(self, flowid):
        self.transfer_variable.set_flowid(flowid)
        LOGGER.debug("Homo split point calculator set flow id: {}".format(flowid))

    def average_run(self, data_instances=None, bin_param: FeatureBinningParam=None, bin_num=10, abnormal_list=None):
        """
        Each client sends split points to server. Then server makes a simple average
        and eliminate duplicate split points

        Parameters
        ----------
        data_instances : DTable
            Input data

        bin_param: FeatureBinningParam
            Setting binning parameters. If it is None, create a default param obj for it.

        bin_num: int, default: 10
            Indicate the max bin nums for binning. If bin_param is provided, it will be ignored.

        abnormal_list: list, default None
            Indicate the values that need to be ignored.
        """
        if bin_param is None:
            bin_param = FeatureBinningParam(bin_num=bin_num)
        if self.role == consts.ARBITER:
            agg_split_points = self._server_run()
        else:
            agg_split_points = self._client_run(data_instances, bin_param=bin_param, abnormal_list=abnormal_list)
        return agg_split_points

    def _client_run(self, data_instances, bin_param, abnormal_list, suffix=tuple()):
        abnormal_detection.empty_table_detection(data_instances)
        abnormal_detection.empty_feature_detection(data_instances)

        if self.bin_method == consts.QUANTILE:
            bin_obj = QuantileBinning(params=bin_param, abnormal_list=abnormal_list, allow_duplicate=True)
        else:
            raise ValueError("Homo Split Point do not accept bin_method: {}".format(self.bin_method))

        split_points = bin_obj.fit_split_points(data_instances)
        if self.role == consts.GUEST:
            self.transfer_variable.guest_split_points.remote(split_points, suffix=suffix)
        else:
            self.transfer_variable.host_split_points.remote(split_points, suffix=suffix)

        agg_split_points = self.transfer_variable.agg_split_points.get(idx=0, suffix=suffix)
        return agg_split_points

    def _server_run(self, suffix=tuple()):
        guest_split_points = self.transfer_variable.guest_split_points.get(idx=0, suffix=suffix)
        host_split_points_list = self.transfer_variable.host_split_points.get(idx=-1, suffix=suffix)

        for host_sp in host_split_points_list:
            assert len(host_sp) == len(guest_split_points)
            assert host_sp.keys() == guest_split_points.keys()

        agg_split_points = {}
        for col_name, s_p in guest_split_points.items():
            agg_split_points[col_name] = [s_p]
            for host_sp in host_split_points_list:
                this_col_sp = host_sp.get(col_name)
                assert len(this_col_sp) == len(s_p)
                agg_split_points[col_name].append(host_sp[col_name])
            col_split_points = agg_split_points[col_name]
            print("col_split_points: {}".format(col_split_points))
            agg_split_points[col_name] = np.mean(col_split_points, axis=0)
        self.transfer_variable.agg_split_points.remote(agg_split_points, idx=-1, role=None, suffix=suffix)
        return agg_split_points






