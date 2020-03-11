from arch.api.utils import log_utils


from federatedml.util import consts
from federatedml.model_base import ModelBase
from federatedml.param.cwjtestparam import CwjParam
from federatedml.feature.homo_feature_binning.homo_split_points import HomoSplitPointCalculator

LOGGER = log_utils.getLogger()

class CWJArbiter(ModelBase):

    def __init__(self):
        super(CWJArbiter, self).__init__()
        self.role = consts.ARBITER
        self.model_param = CwjParam()
        self.binning_obj = HomoSplitPointCalculator(role=self.role)

    def fit(self, train=None, validate=None):
        LOGGER.debug('fit called')
        self.binning_obj.average_run(None, bin_num=10)# Arbiter is None

    def _init_model(self, param):
        LOGGER.info('initialization done')
