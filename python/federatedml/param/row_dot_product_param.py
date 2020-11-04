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

from federatedml.param.base_param import BaseParam
from federatedml.param.encrypt_param import EncryptParam
from federatedml.param.encrypted_mode_calculation_param import EncryptedModeCalculatorParam



class RowDotProductParam(BaseParam):
    """
    Define row dot product parameters

    Parameters
    ----------
    encrypt_param : EncodeParam Object, encrypt method use in row dot product, default: EncryptParam()
    encrypted_mode_calculator_param : EncryptedModeCalculatorParam Object

    """

    def __init__(self, encrypt_param=EncryptParam(), encrypted_mode_calculator_param=EncryptedModeCalculatorParam()):
        self.encrypt_param = encrypt_param
        self.encrypted_mode_calculator_param = encrypted_mode_calculator_param

    def check(self):
        self.encrypt_param.check()
        self.encrypted_mode_calculator_param.check()
        return True
