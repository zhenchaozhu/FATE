from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import BoostingTreeModelMeta
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import QuantileMeta
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import BoostingTreeModelParam

from federatedml.ensemble.boosting.boosting_core import HeteroBoostingHost
from federatedml.param.boosting_tree_param import HeteroSecureBoostParam
from federatedml.ensemble.basic_algorithms import HeteroDecisionTreeHost

from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()


class HeteroSecureBoostHost(HeteroBoostingHost):

    def __init__(self):
        super(HeteroSecureBoostHost, self).__init__()
        self.tree_param = None  # decision tree param
        self.use_missing = False
        self.zero_as_missing = False
        self.cur_epoch_idx = -1
        self.grad_and_hess = None
        self.model_param = HeteroSecureBoostParam()
        self.complete_secure = False

    def _init_model(self, param: HeteroSecureBoostParam):
        super(HeteroSecureBoostHost, self)._init_model(param)
        self.tree_param = param.tree_param
        self.use_missing = param.use_missing
        self.zero_as_missing = param.zero_as_missing
        self.complete_secure = param.complete_secure

    def fit_a_booster(self, epoch_idx: int, booster_dim: int):

        tree = HeteroDecisionTreeHost(tree_param=self.tree_param)
        tree.set_input_data(data_bin=self.data_bin, bin_split_points=self.bin_split_points, bin_sparse_points=
                            self.bin_sparse_points)
        tree.set_valid_features(self.sample_valid_features())
        tree.set_flowid(self.generate_flowid(epoch_idx, booster_dim))
        tree.set_runtime_idx(self.component_properties.local_partyid)

        if self.complete_secure and epoch_idx == 0:
            tree.set_as_complete_secure_tree()

        tree.fit()

        return tree

    def load_booster(self, model_meta, model_param, epoch_idx, booster_idx):
        tree = HeteroDecisionTreeHost(self.tree_param)
        tree.load_model(model_meta, model_param)
        tree.set_flowid(self.generate_flowid(epoch_idx, booster_idx))
        tree.set_runtime_idx(self.component_properties.local_partyid)
        return tree

    def get_model_meta(self):
        model_meta = BoostingTreeModelMeta()
        model_meta.tree_meta.CopyFrom(self.booster_meta)
        model_meta.num_trees = self.boosting_round
        model_meta.quantile_meta.CopyFrom(QuantileMeta(bin_num=self.bin_num))
        meta_name = "HeteroSecureBoostingTreeHostMeta"
        return meta_name, model_meta

    def get_model_param(self):

        model_param = BoostingTreeModelParam()
        model_param.tree_num = len(self.boosting_model_list)
        model_param.tree_dim = self.booster_dim
        model_param.trees_.extend(self.boosting_model_list)
        model_param.feature_name_fid_mapping.update(self.feature_name_fid_mapping)

        param_name = "HeteroSecureBoostingTreeHostParam"

        return param_name, model_param

    def set_model_meta(self, model_meta):
        self.booster_meta = model_meta.tree_meta
        self.boosting_round = model_meta.num_trees
        self.bin_num = model_meta.quantile_meta.bin_num

    def set_model_param(self, model_param):
        self.boosting_model_list = list(model_param.trees_)
        self.booster_dim = model_param.tree_dim
        self.feature_name_fid_mapping.update(model_param.feature_name_fid_mapping)