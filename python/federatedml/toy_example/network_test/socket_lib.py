import pickle
import struct
import socket
import time
from fate_arch.common.profile import profile_logger

# 用于debug代码执行时间
time_map = ['s', 'ms', 'us', 'ns']
def format_time(time_sec):
    for i in range(len(time_map)):
        if time_sec > 1:
            break
        time_sec *= 1000
    res = f'{round(time_sec, 1)}{time_map[i]}'
    return res


class Dbg_Timer():
    def __enter__(self):
        self.start = time.time()
        return self

    def __init__(self, tag, time_th=3):
        self.start = time.time()
        self.tag = tag
        self.time_th = time_th

    def _flied(self):
        return time.time() - self.start

    def flied(self):
        return format_time(self._flied())

    def timeout(self, time_th=None):
        time_th = time_th or self.time_th or 0
        return self._flied() > time_th

    def __exit__(self, *args):
        if self.timeout():
            profile_logger.warning(f'[{self.tag}]: Exec Time={self.flied()}')


class SocketClient:

    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.cs = None
        self.init_client()

    def init_client(self):
        self.cs = self.create_client_communications()

    def create_client_communications(self):
        cs = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        cs.connect((self.ip, self.port))
        return cs

    def send_data(self, data):
        byte_stream = pickle.dumps(data)
        length = len(byte_stream)
        packed_data = struct.pack("LL", 1001, length)
        byte_stream = packed_data + byte_stream
        self.cs.sendall(byte_stream)

    def socket_read_n(self, n):
        buf = b''
        while n > 0:
            data = self.cs.recv(n)
            if data == '':
                raise RuntimeError('unexpected connection close')
            buf += data
            n -= len(data)

        return buf

    def recv_data(self):
        len_buf = self.socket_read_n(16)
        msg_len = struct.unpack('LL', len_buf)[1]
        msg_buf = self.socket_read_n(msg_len)
        return pickle.loads(msg_buf)

    def close(self):
        self.cs.close()


class SocketServer(SocketClient):

    def __init__(self, ip, port, max_client_number=10):
        self.max_client_number = max_client_number
        self.ss = None
        super().__init__(ip, port)

    def init_client(self):
        self.ss = self.create_server_communications()
        cs, accept = self.ss.accept()
        self.cs = cs

    def create_server_communications(self):
        ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ss.bind((self.ip, self.port))
        ss.listen(self.max_client_number)
        return ss

    def close(self):
        self.cs.close()
        self.ss.close()

