from operator import itemgetter
import numpy as np

from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import BoostingTreeModelMeta
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import ObjectiveMeta
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import QuantileMeta
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import BoostingTreeModelParam
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import FeatureImportanceInfo

from federatedml.ensemble.boosting.hetero.hetero_secureboost_host import HeteroSecureBoostHost
from federatedml.param.boosting_tree_param import HeteroFastSecureBoostParam
from federatedml.ensemble.basic_algorithms import HeteroFastDecisionTreeHost
import federatedml.ensemble.boosting.hetero.hetero_fast_secureboost_plan as plan

from arch.api.utils import log_utils

from federatedml.util import consts

LOGGER = log_utils.getLogger()


class HeteroFastSecureBoostHost(HeteroSecureBoostHost):

    def __init__(self):
        super(HeteroFastSecureBoostHost, self).__init__()

        self.k = 1
        self.guest_depth = 0
        self.host_depth = 0
        self.work_mode = consts.MIX_TREE
        self.tree_plan = []
        self.model_param = HeteroFastSecureBoostParam()
        self.model_name = 'fast secureboost'

    def _init_model(self, param: HeteroFastSecureBoostParam):
        super(HeteroFastSecureBoostHost, self)._init_model(param)
        self.k = param.k
        self.work_mode = param.work_mode
        self.guest_depth = param.guest_depth
        self.host_depth = param.host_depth

    def get_tree_plan(self, idx):

        if len(self.tree_plan) == 0:
            self.tree_plan = plan.create_tree_plan(self.work_mode, k=self.k, tree_num=self.boosting_round,
                                                   host_list=self.component_properties.host_party_idlist,
                                                   complete_secure=self.complete_secure)
            LOGGER.debug('tree plan is {}'.format(self.tree_plan))

        return self.tree_plan[idx]

    def fit_a_booster(self, epoch_idx: int, booster_dim: int):

        tree_type, target_host_id = self.get_tree_plan(epoch_idx)

        tree = HeteroFastDecisionTreeHost(tree_param=self.tree_param)
        tree.set_input_data(data_bin=self.data_bin, bin_split_points=self.bin_split_points, bin_sparse_points=
                            self.bin_sparse_points)
        tree.set_valid_features(self.sample_valid_features())
        tree.set_flowid(self.generate_flowid(epoch_idx, booster_dim))
        tree.set_runtime_idx(self.component_properties.local_partyid)
        tree.set_host_party_idlist(self.component_properties.host_party_idlist)

        tree.set_tree_work_mode(tree_type, target_host_id)
        tree.set_layered_depth(self.guest_depth, self.host_depth)
        tree.set_self_host_id(self.component_properties.local_partyid)

        LOGGER.debug('tree work mode is {}'.format(tree_type))
        tree.fit()

        return tree

    def load_booster(self, model_meta, model_param, epoch_idx, booster_idx):

        tree = HeteroFastDecisionTreeHost(self.tree_param)
        tree.load_model(model_meta, model_param)
        tree.set_flowid(self.generate_flowid(epoch_idx, booster_idx))
        tree.set_runtime_idx(self.component_properties.local_partyid)

        if self.tree_plan[epoch_idx][0] == plan.tree_type_dict['guest_feat_only']:
            tree.use_guest_feat_only_predict_mode()

        return tree

    def get_model_meta(self):

        _, model_meta = super(HeteroFastSecureBoostHost, self).get_model_meta()
        meta_name = "HeteroFastSecureBoostHostMeta"
        model_meta.work_mode = self.work_mode
        
        return meta_name, model_meta

    def get_model_param(self):
       
        _, model_param = super(HeteroFastSecureBoostHost, self).get_model_param()
        param_name = "HeteroSecureBoostHostParam"
        model_param.tree_plan.extend(plan.encode_plan(self.tree_plan))

        return param_name, model_param

    def set_model_meta(self, model_meta):
        super(HeteroFastSecureBoostHost, self).set_model_meta(model_meta)
        self.work_mode = model_meta.work_mode

    def set_model_param(self, model_param):
        super(HeteroFastSecureBoostHost, self).set_model_param(model_param)
        self.tree_plan = plan.decode_plan(model_param.tree_plan)
        