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
import numpy as np
from federatedml.feature.instance import Instance
from federatedml.feature.sparse_vector import SparseVector
from arch.api import session


class BaseFilterTest(object):
    def __init__(self):
        self.table_list = []
        self.guest_id = 9999
        self.host_id = 10000

    def _gen_data(self, *args):
        data = []
        shift_iter = 0
        header = [str(i) for i in range(feature_num)]
        # bin_num = 3
        label_count = {}

        bin_num = len(expect_ratio)

        for data_key in range(data_num):
            value = data_key % bin_num
            if value == 0:
                if shift_iter % bin_num == 0:
                    value = bin_num - 1
                shift_iter += 1
            if not is_sparse:
                if not use_random:
                    features = value * np.ones(feature_num)
                else:
                    features = np.random.random(feature_num)
                label = self.__gen_label(value, label_count, expect_ratio)
                inst = Instance(inst_id=data_key, features=features, label=label)

            else:
                if not use_random:
                    features = value * np.ones(feature_num)
                else:
                    features = np.random.random(feature_num)
                data_index = [x for x in range(feature_num)]
                sparse_inst = SparseVector(data_index, data=features, shape=10 * feature_num)
                label = self.__gen_label(value, label_count, expect_ratio)
                inst = Instance(inst_id=data_key, features=sparse_inst, label=label)
                header = [str(i) for i in range(feature_num * 10)]

            data.append((data_key, inst))
        result = session.parallelize(data, include_key=True, partition=partition)
        result.schema = {'header': header}
        self.table_list.append(result)
        return result

    def __gen_label(self, value, label_count: dict, expect_ratio: dict):
        """
        Generate label according to expect event and non-event ratio
        """
        if value not in expect_ratio:
            return np.random.randint(0, 2)

        expect_zero, expect_one = expect_ratio[value]
        if expect_zero == 0:
            return 1
        if expect_one == 0:
            return 0

        if value not in label_count:
            label = 1 if expect_one >= expect_zero else 0
            label_count[value] = [0, 0]
            label_count[value][label] += 1
            return label

        curt_zero, curt_one = label_count[value]
        if curt_zero == 0:
            label_count[value][0] += 1
            return 0
        if curt_one == 0:
            label_count[value][1] += 1
            return 1

        if curt_zero / curt_one <= expect_zero / expect_one:
            label_count[value][0] += 1
            return 0
        else:
            label_count[value][1] += 1
            return 1

    def tearDown(self):
        for table in self.table_list:
            table.destroy()
        print("Finish testing")
