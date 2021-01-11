#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
#  Copyright 2020 The GAIA Authors. All Rights Reserved.
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

from federatedml.param.base_param import BaseParam


class NetworkTestParam(BaseParam):

    def __init__(self, seed=None, use_tcp=False, one_way=True, server_host=None, connect_server_host=None,
                 server_port=None, test_iter=None, block_size=1024, partition=1, data_num=1000):
        self.use_tcp = use_tcp
        self.one_way = one_way
        self.block_size = block_size
        self.server_host = server_host
        self.server_port = server_port
        self.connect_server_host = connect_server_host
        self.test_iter = test_iter

        self.seed = seed
        self.partition = partition
        self.data_num = data_num

    def check(self):
        if self.use_tcp is not None and type(self.use_tcp).__name__ != "bool":
            raise ValueError("Param use_tcp should be None or bool")

        if self.one_way is not None and type(self.one_way).__name__ != "bool":
            raise ValueError("Param one_way should be None or bool")

        if self.block_size is not None and type(self.block_size).__name__ != "int":
            raise ValueError("Param block_size should be None or integers")

        if self.server_host is not None and type(self.server_host).__name__ != "str":
            raise ValueError("Param server_host should be None or string")

        if self.connect_server_host is not None and type(self.connect_server_host).__name__ != "str":
            raise ValueError("Param connect_server_host should be None or string")

        if self.server_port is not None and type(self.server_port).__name__ != "int":
            raise ValueError("Param server_port should be None or integers")

        if self.test_iter is not None and type(self.test_iter).__name__ != "int":
            raise ValueError("Param test_iter should be None or integers")

        if self.seed is not None and type(self.seed).__name__ != "int":
            raise ValueError("random seed should be None or integers")

        if type(self.partition).__name__ != "int" or self.partition < 1:
            raise ValueError("partition should be an integer large than 0")

        if type(self.data_num).__name__ != "int" or self.data_num < 1:
            raise ValueError("data_num should be an integer large than 0")
