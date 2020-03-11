from federatedml.param.base_param import BaseParam

class CwjParam(BaseParam):

    def __init__(self,test=None):
        self.test = test

    def check(self):
        print('hello world')