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

from fate_flow.entity.metric import Metric, MetricMeta
from federatedml.feature.instance import Instance
from federatedml.feature.sparse_vector import SparseVector
from federatedml.param.union_param import UnionParam
from federatedml.model_base import ModelBase
from federatedml.statistic import data_overview

LOGGER = log_utils.getLogger()


class Union(ModelBase):
    def __init__(self):
        super().__init__()
        self.model_param = UnionParam()
        self.metric_name = "union"
        self.metric_namespace = "train"
        self.metric_type = "UNION"

    def _init_model(self, params):
        self.model_param = params
        self.allow_missing = params.allow_missing
        self.is_data_instance = False
        self.is_empty_feature = False
        self.is_sparse = False

    @staticmethod
    def _keep_first(v1, v2):
        return v1

    def check_schema_id(self, local_schema, old_schema):
        if not self.is_data_instance:
            return
        if local_schema.get("sid") != old_schema.get("sid"):
            raise ValueError("Id names do not match! Please check id column names.")

    def check_schema_label_name(self, local_schema, old_schema):
        if not self.is_data_instance:
            return
        local_label_name = local_schema.get("label_name")
        old_label_name = old_schema.get("label_name")
        if local_label_name is None and old_label_name is None:
            return
        if local_label_name is None or old_label_name is None:
            raise ValueError("Union try to combine a labeled data set with an unlabelled one."
                             "Please check labels.")
        if local_label_name != old_label_name:
            raise ValueError("Label names do not match. "
                                 "Please check label column names.")

    def check_schema_header(self, local_schema, old_schema):
        if not self.is_data_instance:
            return
        local_header = local_schema.get("header")
        old_header = old_schema.get("header")
        if local_header != old_header:
            raise ValueError("Table headers do not match! Please check header.")

    def check_feature_length(self, local_table, combined_table):
        if not self.is_data_instance or self.allow_missing:
            return
        combined_table_len = data_overview.get_features_shape(combined_table)
        local_table_len = data_overview.get_features_shape(local_table)
        if  combined_table_len != local_table_len:
            raise ValueError("Feature length {} mismatch length {}.".format(combined_table_len, local_table_len))

    def check_is_data_instance(self, table):
        entry = table.first()
        return isinstance(entry[1], Instance)

    def check_is_sparse(self, table):
        entry = table.first()
        if self.is_data_instance:
            return isinstance(entry[1].features, SparseVector)
        return False

    def check_type_match(self, local_table):
        local_is_data_instance = self.check_is_data_instance(local_table)
        local_is_sparse = self.check_is_sparse(local_table)
        if local_is_data_instance != self.is_data_instance:
            raise ValueError("Type mismatch between DataInstance & non-DataInstance. Union aborted")
        if local_is_sparse != self.is_sparse:
            raise ValueError("Type mismatch between Sparse & non-Sparse tables. Union aborted")

    def fit(self, data):
        if not isinstance(data, dict):
            raise ValueError("Union module must receive more than one table as input.")
        empty_count = 0
        combined_table = None
        combined_schema = None
        metrics = []

        for (key, local_table) in data.items():
            LOGGER.debug("table to combine name: {}".format(key))
            num_data = local_table.count()
            LOGGER.debug("table count: {}".format(num_data))
            local_schema = local_table.schema
            metrics.append(Metric(key, num_data))

            if num_data == 0:
                LOGGER.warning("Table {} is empty.".format(key))
                empty_count += 1
                continue
            if combined_table is None:
                self.is_data_instance = self.check_is_data_instance(local_table)
                self.is_sparse = self.check_is_sparse(local_table)
            if self.is_data_instance:
                self.is_empty_feature = data_overview.is_empty_feature(local_table)
                if self.is_empty_feature:
                    LOGGER.warning("Table {} has empty feature.".format(key))

            if combined_table is None:
                # first table to combine
                combined_table = local_table
                if self.is_data_instance:
                    combined_schema = local_table.schema
                    combined_table.schema = combined_schema

            else:
                self.check_type_match(local_table)
                self.check_schema_id(local_schema, combined_schema)
                self.check_schema_label_name(local_schema, combined_schema)
                self.check_schema_header(local_schema, combined_schema)
                self.check_feature_length(local_table, combined_table)
                old_schema = combined_table.schema
                combined_table = combined_table.union(local_table, self._keep_first)
                combined_table.schema = old_schema


        if combined_table is None:
            num_data = 0
            LOGGER.warning("All tables provided are empty or have empty features.")
        else:
            num_data = combined_table.count()
        metrics.append(Metric("Total", num_data))

        self.callback_metric(metric_name=self.metric_name,
                             metric_namespace=self.metric_namespace,
                             metric_data=metrics)
        self.tracker.set_metric_meta(metric_namespace=self.metric_namespace,
                                     metric_name=self.metric_name,
                                     metric_meta=MetricMeta(name=self.metric_name, metric_type=self.metric_type))

        LOGGER.info("Union operation finished. Total {} empty tables encountered.".format(empty_count))
        return combined_table

    def check_consistency(self):
        pass