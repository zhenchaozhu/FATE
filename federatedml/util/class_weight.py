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


from arch.api import session
from arch.api.utils import log_utils
from federatedml.util import consts

from collections import Counter
import numpy as np


LOGGER = log_utils.getLogger()
session.init("class_weight")


def compute_class_weight(data_instances):
    class_weight  = data_instances.mapPartitions(get_class_weight).reduce(lambda x, y: dict(Counter(x) + Counter(y)))
    n_samples = data_instances.count()
    n_classes = len(class_weight.keys())
    class_weight.update((k, n_samples / (n_classes * v)) for k, v in class_weight.items())

    return class_weight

def get_class_weight(kv_iterator):
    class_dict = {}
    for _, inst in kv_iterator:
        count = class_dict.get(inst.label, 0)
        class_dict[inst.label] = count + 1

    if len(class_dict.keys()) > consts.MAX_CLASSNUM:
        raise ValueError("In Classify Task, max dif classes should no more than %d" % (consts.MAX_CLASSNUM))

    return class_dict

def replace_weight(data_instance, class_weight):
    data_instance.weight = class_weight.get(data_instance.label, 1)
    return data_instance

def compute_sample_weight(class_weight, data_instances):
    return data_instances.mapValues(lambda v: replace_weight(v, class_weight))

def transform(data_instances, class_weight='balanced'):
    if class_weight == 'balanced':
        class_weight = compute_class_weight(data_instances)
    return compute_sample_weight(class_weight, data_instances)

def compute_weight_array(data_instances, class_weight='balanced'):
    if class_weight is None:
        class_weight = {}
    elif class_weight == 'balanced':
        class_weight = compute_class_weight(data_instances)
    weight_inst = data_instances.mapValues(lambda v: class_weight.get(v.label, 1))
    return np.array([v[1] for v in list(weight_inst.collect())])
