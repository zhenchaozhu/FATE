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
################################################################################
#
#
################################################################################

# =============================================================================
# FeatureHistogram
# =============================================================================
import functools
import copy
from arch.api.utils import log_utils
from federatedml.feature.fate_element_type import NoneType
from operator import add, sub
from federatedml.framework.weights import ListWeights, Weights
import numpy as np
from typing import List

LOGGER = log_utils.getLogger()

class HistogramBag(object):

    """
    holds histograms
    """

    def __init__(self, component_num: int = 0, bin_num: list = None, valid_dict: dict = None, hid: int = -1, \
            p_hid: int = -1):

        """
        Parameters
        ----------
        arg1： component_num(integer): number of components in this bag
            or bag(list): list that represents bag content, has structure like:[ [[0,0,0],[0,0,0]...],[[0,0,0]]]

        bin_num: number of bin in every component
        valid_feature: disable a component if {cid:False}
        """

        if bin_num == 0:
            bin_num = []

        self.bin_num = bin_num
        self.component_num = component_num
        self.hid = hid
        self.p_hid = p_hid
        self.bag = []
        component_num = component_num
        assert component_num == len(bin_num)
        for cid in range(component_num):
            if valid_dict is not None and valid_dict[cid] == False:
                self.bag.append([])
            else:
                # self.bag.append(np.zeros((bin_num[fid],3)))
                self.bag.append([[0, 0, 0] for i in range(bin_num[cid])])

    def binary_op(self, other, func, inplace=False):
        assert isinstance(other, HistogramBag)
        assert len(self.bag) == len(other)

        bag = self.bag
        newbag = None
        if not inplace:
            newbag = copy.deepcopy(other)
            bag = newbag.bag

        for bag_idx in range(len(self.bag)):
            for hist_idx in range(len(self.bag[bag_idx])):
                bag[bag_idx][hist_idx][0] = func(self.bag[bag_idx][hist_idx][0], other[bag_idx][hist_idx][0])
                bag[bag_idx][hist_idx][1] = func(self.bag[bag_idx][hist_idx][1], other[bag_idx][hist_idx][1])
                bag[bag_idx][hist_idx][2] = func(self.bag[bag_idx][hist_idx][2], other[bag_idx][hist_idx][2])

        return self if inplace else newbag

    def sub_inplace(self, other):
        self.binary_op(other, sub, inplace=True)

    def add_inplace(self, other):
        self.binary_op(other, add, inplace=True)

    def aggregate_from_left(self):
        for j in range(len(self.bag)):
            for k in range(1, len(self.bag[j])):
                for r in range(len(self.bag[j][k])):
                    self.bag[j][k][r] += self.bag[j][k - 1][r]

    def __add__(self, other):
        return self.binary_op(other, add, inplace=False)

    def __sub__(self, other):
        return self.binary_op(other, sub, inplace=False)

    def __len__(self):
        return len(self.bag)

    def __getitem__(self, item):
        return self.bag[item]

    def __str__(self):
        return str(self.bag)

class FeatureHistogramWeights(Weights):

    def __init__(self, list_of_histogrambags:List[HistogramBag]):

        self.hists = list_of_histogrambags
        super(FeatureHistogramWeights,self).__init__(l=list_of_histogrambags)

    def map_values(self, func, inplace):

        if inplace:
            hists = self.hists
        else:
            hists = copy.deepcopy(self.hists)

        for histbag in hists:
            bag = histbag.bag
            for component_idx in range(len(bag)):
                for hist_idx in range(len(bag[component_idx])):
                    bag[component_idx][hist_idx][0] = func(bag[component_idx][hist_idx][0])
                    bag[component_idx][hist_idx][1] = func(bag[component_idx][hist_idx][1])
                    bag[component_idx][hist_idx][2] = func(bag[component_idx][hist_idx][2])

        if inplace:
            return self
        else:
            return FeatureHistogramWeights(list_of_histogrambags=hists)

    def binary_op(self, other: 'FeatureHistogramWeights', func, inplace:bool):

        new_weights = []
        hists,other_hists = self.hists,other.hists
        for h1,h2 in zip(hists,other_hists):
            rnt = h1.binary_op(h2,func,inplace=inplace)
            if not inplace:
                new_weights.append(rnt)

        if inplace:
            return self
        else:
            return FeatureHistogramWeights(new_weights)

    def axpy(self, a, y: 'FeatureHistogramWeights'):

        func = lambda x1,x2:x1 + a*x2
        self.binary_op(y,func,inplace=True)

        return self

    def __str__(self):
        return str([str(hist) for hist in self.hists])

class FeatureHistogram(object):
    def __init__(self):
        pass

    @staticmethod
    def accumulate_histogram(histograms):
        for i in range(len(histograms)):
            histograms[i].aggregate_from_left()
        return histograms

    @staticmethod
    def calculate_histogram(data_bin, grad_and_hess,
                            bin_split_points, bin_sparse_points,
                            valid_features=None, node_map=None,
                            use_missing=False, zero_as_missing=False):
        LOGGER.info("bin_shape is {}, node num is {}".format(bin_split_points.shape, len(node_map)))
        batch_histogram_cal = functools.partial(
            FeatureHistogram.batch_calculate_histogram,
            bin_split_points=bin_split_points, bin_sparse_points=bin_sparse_points,
            valid_features=valid_features, node_map=node_map,
            use_missing=use_missing, zero_as_missing=zero_as_missing)

        agg_histogram = functools.partial(FeatureHistogram.aggregate_histogram, node_map=node_map)

        batch_histogram = data_bin.join(grad_and_hess, \
                                        lambda data_inst, g_h: (data_inst, g_h)).mapPartitions(batch_histogram_cal)

        return batch_histogram.reduce(agg_histogram)

    @staticmethod
    def aggregate_histogram(batch_histogram1: list, batch_histogram2: list, node_map={}):

        # histogramBag
        for bag1,bag2 in zip(batch_histogram1, batch_histogram2):
            bag1.add_inplace(bag2)
        return batch_histogram1

    @staticmethod
    def batch_calculate_histogram(kv_iterator, bin_split_points=None,
                                  bin_sparse_points=None, valid_features=None,
                                  node_map=None, use_missing=False, zero_as_missing=False):
        data_bins = []
        node_ids = []
        grad = []
        hess = []

        data_record = 0

        for _, value in kv_iterator:
            data_bin, nodeid_state = value[0]
            unleaf_state, nodeid = nodeid_state
            # when conduct hist-subtraction,only compute hist for left nodes
            if unleaf_state == 0 or nodeid not in node_map:
                continue
            g, h = value[1]
            data_bins.append(data_bin)
            node_ids.append(nodeid)
            grad.append(g)
            hess.append(h)

            data_record += 1

        LOGGER.info("begin batch calculate histogram, data count is {}".format(data_record))
        node_num = len(node_map)

        missing_bin = 1 if use_missing else 0

        # node_num, node num * [feature_num], this bag collect g/h sum of every feature in every node
        zero_optim = HistogramBag(node_num, bin_num=[bin_split_points.shape[0]] * node_num)
        zero_opt_node_sum = [[0 for i in range(3)]
                             for j in range(node_num)]

        node_histograms = []
        for k in range(node_num):

            feat_num = bin_split_points.shape[0]
            bin_num = [bin_split_points[fid].shape[0] + 1 + missing_bin for fid in range(feat_num)]
            hist_bag = HistogramBag(feat_num, bin_num, valid_features)
            node_histograms.append(hist_bag)

        if len(node_map) == 0:
            return node_histograms

        for rid in range(data_record):

            nid = node_map.get(node_ids[rid])
            zero_opt_node_sum[nid][0] += grad[rid]
            zero_opt_node_sum[nid][1] += hess[rid]
            zero_opt_node_sum[nid][2] += 1
            for fid, value in data_bins[rid].features.get_all_data():
                if valid_features is not None and valid_features[fid] is False:
                    continue

                if use_missing and value == NoneType():
                    value = -1

                node_histograms[nid][fid][value][0] += grad[rid]
                node_histograms[nid][fid][value][1] += hess[rid]
                node_histograms[nid][fid][value][2] += 1

                zero_optim[nid][fid][0] += grad[rid]
                zero_optim[nid][fid][1] += hess[rid]
                zero_optim[nid][fid][2] += 1

        """
        calculate bin value for sparse points
        """
        for nid in range(node_num):
            for fid in range(bin_split_points.shape[0]):
                if valid_features is not None and valid_features[fid] is True:
                    if not use_missing or (use_missing and not zero_as_missing):
                        sparse_point = bin_sparse_points[fid]
                        node_histograms[nid][fid][sparse_point][0] += zero_opt_node_sum[nid][0] - zero_optim[nid][fid][0]
                        node_histograms[nid][fid][sparse_point][1] += zero_opt_node_sum[nid][1] - zero_optim[nid][fid][1]
                        node_histograms[nid][fid][sparse_point][2] += zero_opt_node_sum[nid][2] - zero_optim[nid][fid][2]
                    else:
                        node_histograms[nid][fid][-1][0] += zero_opt_node_sum[nid][0] - zero_optim[nid][fid][0]
                        node_histograms[nid][fid][-1][1] += zero_opt_node_sum[nid][1] - zero_optim[nid][fid][1]
                        node_histograms[nid][fid][-1][2] += zero_opt_node_sum[nid][2] - zero_optim[nid][fid][2]

        LOGGER.info("batch compute done")

        return node_histograms