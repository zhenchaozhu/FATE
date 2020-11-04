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


from federatedml.model_base import ModelBase
from federatedml.param.row_dot_product_param import RowDotProductParam
from federatedml.secureprotol import PaillierEncrypt
from federatedml.secureprotol.encrypt_mode import EncryptModeCalculator
from federatedml.transfer_variable.transfer_class.row_dot_product_transfer_variable import \
    RowDotProductTransferVariable
from federatedml.util import consts, fate_operator, LOGGER


class RowDotProductBase(ModelBase):
    def __init__(self):
        super().__init__()
        self.model_param = RowDotProductParam()
        self.model_name = 'RowDotProduct'
        # self.model_param_name = 'RowDotProductModelParam'
        # self.model_meta_name = 'RowDotProductModelMeta'
        self.transfer_variable = RowDotProductTransferVariable()
        self.encrypter = None
        self.encrypted_calculator = None

    def _init_model(self, params):
        self.model_param = params
        self.method = params.encrypt_param.method
        self.key_length = params.encrypt_param.key_length
        self.re_encrypt_rate = params.encrypted_mode_calculator_param
        self.mode = params.encrypted_mode_calculator_param.mode
        self.re_encrypted_rate = params.encrypted_mode_calculator_param.re_encrypted_rate

    def generate_encrypter(self):
        LOGGER.info("generate encrypter")
        if self.method.lower() == consts.PAILLIER.lower():
            self.encrypter = PaillierEncrypt()
            self.encrypter.generate_key(self.key_length)
        else:
            raise NotImplementedError("encrypt method not supported yes!!!")

        self.encrypted_calculator = EncryptModeCalculator(self.encrypter, self.mode, self.re_encrypted_rate)

    def fit(self, data_instances):
        raise ValueError(f"RowDotProductBase `fit` not implemented.")


class RowDotProductGuest(RowDotProductBase):
    def __init__(self):
        super().__init__()
        self.role = consts.GUEST
        self.guest_forward_transfer = self.transfer_variable.guest_forward
        self.product_forward_transfer = self.transfer_variable.product_forward

    def fit(self, data_instances):
        LOGGER.info(f"Guest start Row Dot Product, encrypt method: {self.method}")

        self.generate_encrypter()
        encrypter, encrypted_calculator = self.encrypter, self.encrypted_calculator
        guest_forward = data_instances.mapValues(lambda x: x.features)
        guest_forward = encrypted_calculator.encrypt(guest_forward)
        self.guest_forward_transfer.remote(obj=guest_forward, role=consts.HOST, idx=0)
        product_forward = self.product_forward_transfer.get(idx=0)
        result = product_forward.mapValues(lambda x: encrypter.decrypt(x))

        schema = data_instances.schema
        result_schema = {"header": ["result"],
                         "sid_name": schema.get('sid_name')}
        result.schema = result_schema

        LOGGER.info(f"Finish Guest Row Dot Product!")

        return result


class RowDotProductHost(RowDotProductBase):
    def __init__(self):
        super().__init__()
        self.role = consts.HOST
        self.guest_forward_transfer = self.transfer_variable.guest_forward
        self.product_forward_transfer = self.transfer_variable.product_forward

    def fit(self, data_instances):
        LOGGER.info(f"Host start Row Dot Product, encrypt method: {self.method}")

        guest_forward = self.guest_forward_transfer.get(idx=0)
        product_forward = data_instances.join(guest_forward, lambda h, g: fate_operator.dot(h, g))
        self.product_forward_transfer.remote(obj=product_forward, role=consts.GUEST, idx=0)

        LOGGER.info(f"Finish Host Row Dot Product!")

        return data_instances
