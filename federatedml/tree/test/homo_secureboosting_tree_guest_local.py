from federatedml.tree.test.homo_secureboosting_client_local import HomoSecureBoostingTreeClient
from federatedml.util import consts
from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()


class HomoSecureBoostingTreeGuest(HomoSecureBoostingTreeClient):

    def __init__(self):
        super(HomoSecureBoostingTreeGuest, self).__init__()
        self.role = consts.GUEST







