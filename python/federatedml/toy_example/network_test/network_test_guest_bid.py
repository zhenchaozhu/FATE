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
import os
import sys
import math
import numpy as np
# from arch.session import computing_session as session
from federatedml.model_base import ModelBase
from federatedml.param.network_test_param import NetworkTestParam
from federatedml.transfer_variable.transfer_class.network_test_transfer_variable import NetworkTestTransferVariable
from federatedml.util.param_extract import ParamExtract
from federatedml.util import LOGGER
import time
from federatedml.toy_example.network_test import socket_lib
from federatedml.toy_example.network_test.socket_lib import Dbg_Timer

def _assert(a, b, desc=''):
    if isinstance(a, np.ndarray):
        assert (a == b).all(), desc
    else:
        assert a == b, desc


class NetworkTestGuest(ModelBase):
    def __init__(self):
        super(NetworkTestGuest, self).__init__()
        self.data_num = None
        self.partition = None
        self.seed = None
        self.transfer_inst = NetworkTestTransferVariable()
        self.model_param = NetworkTestParam()
        self.max_client_number = 10
        self.cs = None

    def _init_runtime_parameters(self, component_parameters):
        param_extractor = ParamExtract()
        param = param_extractor.parse_param_from_config(self.model_param, component_parameters)
        self._init_model(param)
        return param

    def _init_model(self, model_param):
        self.data_num = model_param.data_num
        self.partition = model_param.partition
        self.seed = model_param.seed
        self.use_tcp = model_param.use_tcp
        self.block_size = model_param.block_size
        self.server_host = model_param.server_host
        self.server_port = model_param.server_port
        self.connect_server_host = model_param.connect_server_host

    def test_net_work(self, socket_client, send_str, loop, max_iter=200):
        LOGGER.info("-------> bid_way")
        with Dbg_Timer(f"bid_way-TCP", 0):
            for i in range(max_iter):
                socket_client.send_data(send_str)
                _data = socket_client.recv_data()
                _assert(send_str, _data, f'{_data}')

        with Dbg_Timer(f"bid_way-CUBE", 0):
            for i in range(max_iter):
                self.transfer_inst.guest_share.remote(send_str, role='host', idx=0, suffix="nw1_%s_%s" % (i, loop))
                _data = self.transfer_inst.host_share.get(idx=0, suffix="nw1_%s_%s" % (i, loop))
                _assert(send_str, _data, f'{_data}')

    def run(self, component_parameters=None, args=None):
        self._init_runtime_parameters(component_parameters)
        max_iter = self.model_param.test_iter

        time.sleep(3)
        LOGGER.info("begin to test guest network")
        socket_client = socket_lib.SocketClient(self.connect_server_host, self.server_port)
        try:
            LOGGER.info(f">>>>>>>>>>>>>>>>>>>> Start to test Perf.")
            iter = 1
            base_str = 'a'

            send_str = base_str
            LOGGER.info(f">>>>>>>>> test send size is 1 Bytes")
            self.test_net_work(socket_client, send_str, iter + 1, max_iter=max_iter)

            send_str = base_str * 1024
            LOGGER.info(f">>>>>>>>> test send size is 1KB")
            self.test_net_work(socket_client, send_str, iter + 2, max_iter=max_iter)

            send_str = base_str * 1024 * 10
            LOGGER.info(f">>>>>>>>> test send size is 10KB")
            self.test_net_work(socket_client, send_str, iter + 3, max_iter=max_iter)

            send_str = base_str * 1024 * 10 * 10
            LOGGER.info(f">>>>>>>>> test send size is 100KB")
            self.test_net_work(socket_client, send_str, iter + 4, max_iter=max_iter)

            send_str = base_str * 1024 * 1024
            LOGGER.info(f">>>>>>>>> test send size is 1MB")
            self.test_net_work(socket_client, send_str, iter + 5, max_iter=10)

            send_str = base_str * 1024 * 1024 * 10
            LOGGER.info(f">>>>>>>>> test send size is 10MB")
            self.test_net_work(socket_client, send_str, iter + 6, max_iter=10)

            send_str = base_str * 1024 * 1024 * 100
            LOGGER.info(f">>>>>>>>> test send size is 100MB")
            self.test_net_work(socket_client, send_str, iter + 7, max_iter=2)
        except Exception as e:
            LOGGER.error(f"network raise exception {e}")
            raise
        finally:
            socket_client.close()
