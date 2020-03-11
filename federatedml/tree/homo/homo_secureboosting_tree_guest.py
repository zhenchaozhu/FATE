from federatedml.tree import HomoSecureBoostingTreeClient
from federatedml.tree import SecureBoostClientAggregator

from federatedml.util import consts
from arch.api.utils import log_utils

from federatedml.feature.homo_feature_binning.homo_split_points import HomoSplitPointCalculator

LOGGER = log_utils.getLogger()


class HomoSecureBoostingTreeGuest(HomoSecureBoostingTreeClient):

    def __init__(self):
        super(HomoSecureBoostingTreeGuest, self).__init__()
        self.role = consts.GUEST
        self.aggregator = SecureBoostClientAggregator(transfer_variable=self.transfer_inst, role=self.role)
        self.binning_obj = HomoSplitPointCalculator(role=self.role, )








