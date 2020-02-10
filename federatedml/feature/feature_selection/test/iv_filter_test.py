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

import argparse
import math

import numpy as np

from arch.api import federation
from arch.api import session
from federatedml.feature.feature_selection import iv_value_select_filter, iv_percentile_filter
from federatedml.feature.feature_selection.selection_properties import SelectionProperties
from federatedml.feature.feature_selection.test.feature_selection_test import BaseFilterTest
from federatedml.feature.instance import Instance
from federatedml.feature.test.test_quantile_binning_module import hetero_feature_binning_test
from federatedml.param.feature_selection_param import IVValueSelectionParam, IVPercentileSelectionParam
from federatedml.transfer_variable.transfer_class.hetero_feature_selection_transfer_variable import \
    HeteroFeatureSelectionTransferVariable

GUEST = 'guest'
HOST = 'host'


class IvFilterTest(BaseFilterTest):
    def __init__(self, role):
        super().__init__()
        self.role = role
        self.filter_obj = None
        self.binning_test_obj = None
        self.job_id = ''

    def setup_session(self, job_id, guest_id, host_id):
        self.guest_id = guest_id
        self.host_id = host_id
        self.job_id = job_id
        session.init(job_id)
        federation.init(job_id,
                        {"local": {
                            "role": self.role,
                            "party_id": guest_id if self.role == GUEST else host_id
                        },
                            "role": {
                                "host": [
                                    host_id
                                ],
                                "guest": [
                                    guest_id
                                ]
                            }
                        })

    def setup_iv_value_obj(self, iv_param: IVValueSelectionParam or IVPercentileSelectionParam,
                           selection_properties: SelectionProperties,
                           obj_type='iv_value'):
        if obj_type == 'iv_value':
            my_filter = iv_value_select_filter
        else:
            my_filter = iv_percentile_filter

        if self.role == GUEST:
            filter_obj = my_filter.Guest(iv_param)
        else:
            filter_obj = my_filter.Host(iv_param)

        filter_obj.set_selection_properties(selection_properties)
        if self.binning_test_obj is None:
            binning_test_obj = hetero_feature_binning_test.TestHeteroFeatureBinning(self.role,
                                                                                    self.guest_id, self.host_id)
            self.binning_test_obj = binning_test_obj
        else:
            binning_test_obj = self.binning_test_obj
        return filter_obj, binning_test_obj

    def test_iv_value_filter(self):
        iv_param = IVValueSelectionParam(value_threshold=0.1)
        selection_properties = SelectionProperties()
        # data_instances = self._gen_fix_data(1000)
        data_event_counts = [[(10, 90), (30, 70), (50, 50), (70, 30), (90, 10)],
                             [(20, 80), (40, 60), (50, 50), (60, 40), (80, 20)],
                             [(45, 55), (47, 53), (50, 50), (53, 47), (45, 55)],
                             [(50, 50), (50, 50), (50, 50), (50, 50), (50, 50)]]
        data_instances = self._gen_data(data_event_counts)

        selection_properties.set_header(data_instances.schema['header'])
        selection_properties.set_select_all_cols()
        selection_properties.set_last_left_col_indexes([x for x in range(len(data_event_counts))])

        filter_obj, binning_test_obj = self.setup_iv_value_obj(iv_param, selection_properties)
        table_args = {"data": {'HeteroFeatureBinning': {"data": data_instances}}}
        binning_obj = binning_test_obj.run_data(table_args, 'fit')
        # for col_name, col_result in binning_obj.binning_obj.bin_results.all_cols_results.items():
        #     print("col_name: {}, binning_obj iv: {}".format(col_name, col_result.iv))
        transfer_variable = HeteroFeatureSelectionTransferVariable()
        transfer_variable.set_flowid(self.job_id + 'iv_value')

        filter_obj.set_binning_obj(binning_obj)
        filter_obj.set_transfer_variable(transfer_variable)
        filter_obj.fit(data_instances, suffix='iv_value')

        result_selection_properties = filter_obj.selection_properties
        # for fid, d_event_count in enumerate(data_event_counts):
        #     print("fid: {}, iv: {}".format(fid, self.iv_cals(d_event_count)))
        """fid: 0, iv: 1.6773590448190858
        fid: 1, iv: 0.7302957106348537
        fid: 2, iv: 0.015423798428087228
        fid: 3, iv: 0.0
        """
        assert result_selection_properties.left_col_names == ['d0', 'd1']

    def test_iv_percentile_filter(self):
        iv_param = IVPercentileSelectionParam(percentile_threshold=0.8)
        selection_properties = SelectionProperties()
        data_event_counts = [[(10, 90), (30, 70), (50, 50), (70, 30), (90, 10)],
                             [(20, 80), (40, 60), (50, 50), (60, 40), (80, 20)],
                             [(45, 55), (47, 53), (50, 50), (53, 47), (45, 55)],
                             [(50, 50), (50, 50), (50, 50), (50, 50), (50, 50)]]
        data_instances = self._gen_data(data_event_counts)
        filter_obj, binning_test_obj = self.setup_iv_value_obj(iv_param, selection_properties, obj_type='iv_percentile')
        table_args = {"data": {'HeteroFeatureBinning': {"data": data_instances}}}
        binning_obj = binning_test_obj.run_data(table_args, 'fit')

        selection_properties.set_header(data_instances.schema['header'])
        selection_properties.set_select_all_cols()
        filter_obj.set_binning_obj(binning_obj)

        transfer_variable = HeteroFeatureSelectionTransferVariable()
        transfer_variable.set_flowid(self.job_id + 'iv_percentile')

        filter_obj.set_transfer_variable(transfer_variable)
        filter_obj.set_selection_properties(selection_properties)

        filter_obj.fit(data_instances, suffix='iv_percentile')
        result_selection_properties = filter_obj.selection_properties

        print("result_selection_properties: {}".format(result_selection_properties.left_col_names))

    def _gen_fix_data(self, data_num=100):
        bin_nums = 10
        label = [1] * (data_num // 2) + [0] * (data_num // 2)
        data_nums_in_bin = data_num // bin_nums
        d1 = []
        for k in range(bin_nums):
            d1 += [k] * data_nums_in_bin
        d2 = [i % bin_nums for i in range(data_num)]
        d3_tail = []
        d3 = []
        for idx, d in enumerate(d1):
            if idx % bin_nums < int(bin_nums * 0.9):
                d3.append(d)
            else:
                d3_tail.append(d)
        d3 += d3_tail
        assert len(label) == len(d1) == len(d2) == len(d3)
        data = []
        for k in range(len(label)):
            features = np.array([d1[k], d2[k], d3[k]])
            this_label = label[k]
            inst = Instance(inst_id=k, features=features, label=this_label)
            data.append((k, inst))
        result = session.parallelize(data, include_key=True, partition=48)
        result.schema = {'header': ['d1', 'd2', 'd3']}
        self.table_list.append(result)
        return result

    def _gen_data(self, data_event_counts):
        """
        Generate data according to provided data event count
        Parameters
        ----------
        data_event_counts : list of list
            each element indicate event count and non-event count distribution
            [  [(event_sum, non-event_sum), (same sum in second_bin), (in third bin) ...],
             [distribution in second feature],
             ...
             ]
        """
        total_ones = 0
        total_zeros = 0
        for event_sum, non_event_sum in data_event_counts[0]:
            total_ones += event_sum
            total_zeros += non_event_sum
        data_num = total_ones + total_zeros
        feature_num = len(data_event_counts)

        features = []
        for event_counts in data_event_counts:
            # print('event_counts: {}, iv: {}'.format(event_counts, self.iv_cals(event_counts)))
            k = len(event_counts)
            values = []
            for idx, (_, non_event_sum) in enumerate(event_counts):
                values.extend([idx - k // 2] * non_event_sum)
            for idx, (event_sum, _) in enumerate(event_counts):
                values.extend([idx - k // 2] * event_sum)
            features.append(values)
        # print("before transpose: {}".format(features))
        features = np.transpose(np.array(features))
        # print("after transpose: {}".format(features))

        assert features.shape[0] == data_num

        data_insts = []
        for data_idx in range(data_num):
            if data_idx < total_zeros:
                label = 0
            else:
                label = 1
            inst = Instance(inst_id=data_idx, features=features[data_idx], label=label)
            data_insts.append((data_idx, inst))
        result = session.parallelize(data_insts, include_key=True, partition=48)
        result.schema = {'header': ['d' + str(x) for x in range(feature_num)]}
        self.table_list.append(result)
        return result

    @staticmethod
    def iv_cals(data_event_count, adjustment_factor=0.5):
        event_total = 0
        non_event_total = 0
        for event_sum, non_event_sum in data_event_count:
            event_total += event_sum
            non_event_total += non_event_sum

        iv = 0
        for event_count, non_event_count in data_event_count:
            if event_count == 0 or non_event_count == 0:
                event_rate = 1.0 * (event_count + adjustment_factor) / event_total
                non_event_rate = 1.0 * (non_event_count + adjustment_factor) / non_event_total
            else:
                event_rate = 1.0 * event_count / event_total
                non_event_rate = 1.0 * non_event_count / non_event_total
            woe_i = math.log(non_event_rate / event_rate)
            iv_i = (non_event_rate - event_rate) * woe_i
            iv += iv_i
        return iv


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--role', required=False, type=str, help="role",
                        choices=(GUEST, HOST), default=GUEST)
    parser.add_argument('-gid', '--gid', required=False, type=str, help="guest party id", default='9999')
    parser.add_argument('-hid', '--hid', required=False, type=str, help="host party id", default='10000')
    parser.add_argument('-j', '--job_id', required=True, type=str, help="job_id")

    args = parser.parse_args()

    args = parser.parse_args()
    job_id = args.job_id
    guest_id = args.gid
    host_id = args.hid
    role = args.role

    test_obj = IvFilterTest(role)
    test_obj.setup_session(job_id, guest_id, host_id)
    test_obj.test_iv_value_filter()
    test_obj.test_iv_percentile_filter()
    test_obj.tearDown()
