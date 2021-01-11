from federatedml.model_base import ModelBase
from federatedml.param.network_test_param import NetworkTestParam
from federatedml.transfer_variable.transfer_class.network_test_transfer_variable import NetworkTestTransferVariable
from federatedml.util.param_extract import ParamExtract
from federatedml.util import LOGGER
from federatedml.toy_example.network_test import socket_lib
import numpy as np
from federatedml.toy_example.network_test.socket_lib import Dbg_Timer


def _assert(a, b, desc=''):
    if isinstance(a, np.ndarray):
        assert (a == b).all(), desc
    else:
        assert a == b, desc


class NetworkTestHost(ModelBase):
    def __init__(self):
        super(NetworkTestHost, self).__init__()
        self.transfer_inst = NetworkTestTransferVariable()
        self.model_param = NetworkTestParam()
        self.data_output = None
        self.model_output = None
        self.max_client_number = 10

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

    def test_net_work(self, socket_server, send_str, loop, max_iter=200):
        LOGGER.info("-------> bid_way")
        # test send & recv
        with Dbg_Timer(f"bid_way-TCP", 0):
            for i in range(max_iter):
                _data = socket_server.recv_data()
                _assert(send_str, _data, f'{_data}')
                socket_server.send_data(send_str)

        with Dbg_Timer(f"bid_way-CUBE", 0):
            for i in range(max_iter):
                _data = self.transfer_inst.guest_share.get(idx=0, suffix="nw1_%s_%s" % (i, loop))
                _assert(send_str, _data, f'{_data}')
                self.transfer_inst.host_share.remote(send_str, role="guest", idx=0, suffix="nw1_%s_%s" % (i, loop))

    def run(self, component_parameters=None, args=None):
        LOGGER.info("begin to init parameters of secure add example host")
        self._init_runtime_parameters(component_parameters)
        socket_server = socket_lib.SocketServer(self.server_host, self.server_port)
        max_iter = self.model_param.test_iter

        LOGGER.info("begin to test host network")
        try:
            iter = 1
            LOGGER.info(f">>>>>>>>>>>>>>>>>>>> Start to test Perf.")
            base_str = 'a'
            send_str = base_str
            LOGGER.info(f">>>>>>>>> test send size is 1 Bytes")

            self.test_net_work(socket_server, send_str, iter + 1, max_iter=max_iter)

            send_str = base_str * 1024
            LOGGER.info(f">>>>>>>>> test send size is 1KB")
            self.test_net_work(socket_server, send_str, iter + 2, max_iter=max_iter)

            send_str = base_str * 1024 * 10
            LOGGER.info(f">>>>>>>>> test send size is 10KB")
            self.test_net_work(socket_server, send_str, iter + 3, max_iter=max_iter)

            send_str = base_str * 1024 * 10 * 10
            LOGGER.info(f">>>>>>>>> test send size is 100KB")
            self.test_net_work(socket_server, send_str, iter + 4, max_iter=max_iter)

            send_str = base_str * 1024 * 1024
            LOGGER.info(f">>>>>>>>> test send size is 1MB")
            self.test_net_work(socket_server, send_str, iter + 5, max_iter=10)

            send_str = base_str * 1024 * 1024 * 10
            LOGGER.info(f">>>>>>>>> test send size is 10MB")
            self.test_net_work(socket_server, send_str, iter + 6, max_iter=10)

            send_str = base_str * 1024 * 1024 * 100
            LOGGER.info(f">>>>>>>>> test send size is 100MB")
            self.test_net_work(socket_server, send_str, iter + 7, max_iter=2)
        except Exception as e:
            LOGGER.error(f"network raise exception {e}")
            raise
        finally:
            socket_server.close()
