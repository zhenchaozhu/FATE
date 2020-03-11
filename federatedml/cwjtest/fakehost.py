from federatedml.cwjtest.basemodel import CWJBase
from federatedml.util import consts
from federatedml.feature.homo_feature_binning.homo_split_points import HomoSplitPointCalculator

class FakeHost(CWJBase):

    def __init__(self):
        super(FakeHost, self).__init__()
        self.role = consts.HOST
        binning_obj = HomoSplitPointCalculator(role=self.role, )
        self.binning_obj = binning_obj