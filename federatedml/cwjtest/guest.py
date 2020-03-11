from federatedml.cwjtest.basemodel import CWJBase
from federatedml.util import consts
from federatedml.feature.homo_feature_binning.homo_split_points import HomoSplitPointCalculator

class CWJGuest(CWJBase):

    def __init__(self):
        super(CWJGuest, self).__init__()
        self.role = consts.GUEST
        binning_obj = HomoSplitPointCalculator(role=self.role)
        self.binning_obj = binning_obj