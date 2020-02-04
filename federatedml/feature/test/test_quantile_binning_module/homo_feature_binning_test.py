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

import numpy as np

from arch.api import federation
from arch.api import session
from federatedml.feature.homo_feature_binning.homo_split_points import HomoSplitPointCalculator
from federatedml.feature.instance import Instance
from federatedml.feature.sparse_vector import SparseVector

GUEST = 'guest'
HOST = 'host'

host_id_list = ['10000', '10001', '10002']


class TestHomoFeatureBinning():
    def __init__(self, role, own_id):
        self.role = role
        self.party_id = own_id
        self.model_name = 'HomoFeatureBinning'
        self.args = None
        self.table_list = []

    def _gen_data(self, data_num, feature_num, partition, expect_split_points, is_sparse=False, use_random=False):
        data = []
        shift_iter = 0
        header = [str(i) for i in range(feature_num)]
        bin_num = len(expect_split_points)

        for data_key in range(data_num):
            value = expect_split_points[data_key % bin_num]
            if value == expect_split_points[0]:
                if shift_iter % bin_num == 0:
                    value = expect_split_points[bin_num - 1]
                shift_iter += 1
            if not is_sparse:
                if not use_random:
                    features = value * np.ones(feature_num)
                else:
                    features = np.random.random(feature_num)
                inst = Instance(inst_id=data_key, features=features, label=data_key % 2)

            else:
                if not use_random:
                    features = value * np.ones(feature_num)
                else:
                    features = np.random.random(feature_num)
                data_index = [x for x in range(feature_num)]
                sparse_inst = SparseVector(data_index, data=features, shape=10 * feature_num)
                inst = Instance(inst_id=data_key, features=sparse_inst, label=data_key % 2)
                header = [str(i) for i in range(feature_num * 10)]

            data.append((data_key, inst))
        result = session.parallelize(data, include_key=True, partition=partition)
        result.schema = {'header': header}
        self.table_list.append(result)
        return result

    def test_homo_split_points(self):
        binning_obj = HomoSplitPointCalculator(role=self.role)
        guest_split_points = (1, 2, 3)
        host_split_points = [(4, 5, 6), (7, 8, 9), (10, 11, 12)]
        expect_agg_sp = [guest_split_points]
        expect_agg_sp.extend(host_split_points)
        expect_agg_sp = np.mean(expect_agg_sp, axis=0)
        if self.role == GUEST:
            data_inst = self._gen_data(1000, 10, 48, expect_split_points=guest_split_points)
        elif self.role == 'arbiter':
            data_inst = None
        else:
            host_idx = host_id_list.index(self.party_id)
            data_inst = self._gen_data(1000, 10, 48, expect_split_points=host_split_points[host_idx])
        agg_sp = binning_obj.average_run(data_inst)
        for col_name, col_agg_sp in agg_sp.items():
            assert np.all(col_agg_sp == expect_agg_sp)
        return

    def tearDown(self):
        for table in self.table_list:
            table.destroy()
        print("Finish testing")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--role', required=False, type=str, help="role",
                        choices=(GUEST, HOST, 'arbiter'), default=GUEST)
    parser.add_argument('-pid', '--pid', required=True, type=str, help="own party id")
    parser.add_argument('-j', '--job_id', required=True, type=str, help="job_id")

    args = parser.parse_args()
    job_id = args.job_id
    own_party_id = args.pid
    role = args.role
    print("args: {}".format(args))
    session.init(job_id)
    federation.init(job_id,
                    {"local": {
                        "role": role,
                        "party_id": own_party_id
                    },
                        "role": {
                            "host": [str(x) for x in host_id_list],
                            "guest": [
                                '9999'
                            ],
                            "arbiter": ['9998']
                        }
                    })

    test_obj = TestHomoFeatureBinning(role, own_party_id)
    # homo_obj.test_homo_lr()
    test_obj.test_homo_split_points()
    test_obj.tearDown()
