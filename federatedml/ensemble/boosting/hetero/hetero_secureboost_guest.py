from operator import itemgetter
import numpy as np

from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import BoostingTreeModelMeta
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import ObjectiveMeta
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import QuantileMeta
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import BoostingTreeModelParam
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import FeatureImportanceInfo

from federatedml.ensemble.boosting.boosting_core import HeteroBoostingGuest
from federatedml.param.boosting_tree_param import HeteroSecureBoostParam
from federatedml.ensemble.basic_algorithms import HeteroDecisionTreeGuest
from federatedml.util import consts

from arch.api.table.eggroll.table_impl import DTable

from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()


class HeteroSecureBoostGuest(HeteroBoostingGuest):

    def __init__(self):
        super(HeteroSecureBoostGuest, self).__init__()
        self.tree_param = None  # decision tree param
        self.use_missing = False
        self.zero_as_missing = False
        self.cur_epoch_idx = -1
        self.grad_and_hess = None
        self.feature_importances_ = {}
        self.model_param = HeteroSecureBoostParam()
        self.complete_secure = False

    def _init_model(self, param: HeteroSecureBoostParam):
        super(HeteroSecureBoostGuest, self)._init_model(param)
        self.tree_param = param.tree_param
        self.use_missing = param.use_missing
        self.zero_as_missing = param.zero_as_missing
        self.complete_secure = param.complete_secure

    def compute_grad_and_hess(self, y_hat: DTable, y: DTable):
        LOGGER.info("compute grad and hess")
        loss_method = self.loss
        if self.task_type == consts.CLASSIFICATION:
            grad_and_hess = y.join(y_hat, lambda y, f_val: \
                (loss_method.compute_grad(y, loss_method.predict(f_val)), \
                 loss_method.compute_hess(y, loss_method.predict(f_val))))
        else:
            grad_and_hess = y.join(y_hat, lambda y, f_val:
            (loss_method.compute_grad(y, f_val),
             loss_method.compute_hess(y, f_val)))

        return grad_and_hess

    @staticmethod
    def get_grad_and_hess(g_h: DTable, dim=0):
        LOGGER.info("get grad and hess of tree {}".format(dim))
        grad_and_hess_subtree = g_h.mapValues(
            lambda grad_and_hess: (grad_and_hess[0][dim], grad_and_hess[1][dim]))
        return grad_and_hess_subtree

    def update_feature_importance(self, tree_feature_importance):
        for fid in tree_feature_importance:
            if fid not in self.feature_importances_:
                self.feature_importances_[fid] = 0

            self.feature_importances_[fid] += tree_feature_importance[fid]

    def fit_a_booster(self, epoch_idx: int, booster_dim: int):

        if self.cur_epoch_idx != epoch_idx:
            self.grad_and_hess = self.compute_grad_and_hess(self.y_hat, self.y)
            self.cur_epoch_idx = epoch_idx

        g_h = self.get_grad_and_hess(self.grad_and_hess, booster_dim)

        tree = HeteroDecisionTreeGuest(tree_param=self.tree_param)
        tree.set_input_data(self.data_bin, self.bin_split_points, self.bin_sparse_points)
        tree.set_grad_and_hess(g_h)
        tree.set_encrypter(self.encrypter)
        tree.set_encrypted_mode_calculator(self.encrypted_calculator)
        tree.set_valid_features(self.sample_valid_features())
        tree.set_flowid(self.generate_flowid(epoch_idx, booster_dim))
        tree.set_host_party_idlist(self.component_properties.host_party_idlist)
        tree.set_runtime_idx(self.component_properties.local_partyid)

        if self.cur_epoch_idx == 0 and self.complete_secure:
            tree.set_as_complete_secure_tree()

        tree.fit()

        self.update_feature_importance(tree.get_feature_importance())

        return tree

    def load_booster(self, model_meta, model_param, epoch_idx, booster_idx):
        tree = HeteroDecisionTreeGuest(self.tree_param)
        tree.load_model(model_meta, model_param)
        tree.set_flowid(self.generate_flowid(epoch_idx, booster_idx))
        tree.set_runtime_idx(self.component_properties.local_partyid)
        tree.set_host_party_idlist(self.component_properties.host_party_idlist)
        return tree

    def get_model_meta(self):
        model_meta = BoostingTreeModelMeta()
        model_meta.tree_meta.CopyFrom(self.booster_meta)
        model_meta.learning_rate = self.learning_rate
        model_meta.num_trees = self.boosting_round
        model_meta.quantile_meta.CopyFrom(QuantileMeta(bin_num=self.bin_num))
        model_meta.objective_meta.CopyFrom(ObjectiveMeta(objective=self.objective_param.objective,
                                                         param=self.objective_param.params))
        model_meta.task_type = self.task_type
        model_meta.n_iter_no_change = self.n_iter_no_change
        model_meta.tol = self.tol
        meta_name = "HeteroSecureBoostingTreeGuestMeta"

        return meta_name, model_meta

    def get_model_param(self):

        model_param = BoostingTreeModelParam()
        model_param.tree_num = len(self.boosting_model_list)
        model_param.tree_dim = self.booster_dim
        model_param.trees_.extend(self.boosting_model_list)
        model_param.init_score.extend(self.init_score)
        model_param.losses.extend(self.history_loss)
        model_param.classes_.extend(map(str, self.classes_))
        model_param.num_classes = self.num_classes

        feature_importances = list(self.feature_importances_.items())
        feature_importances = sorted(feature_importances, key=itemgetter(1), reverse=True)
        feature_importance_param = []
        for (sitename, fid), _importance in feature_importances:
            feature_importance_param.append(FeatureImportanceInfo(sitename=sitename,
                                                                  fid=fid,
                                                                  importance=_importance))
        model_param.feature_importances.extend(feature_importance_param)

        model_param.feature_name_fid_mapping.update(self.feature_name_fid_mapping)

        param_name = "HeteroSecureBoostingTreeGuestParam"

        return param_name, model_param

    def set_model_meta(self, model_meta):
        self.booster_meta = model_meta.tree_meta
        self.learning_rate = model_meta.learning_rate
        self.boosting_round = model_meta.num_trees
        self.bin_num = model_meta.quantile_meta.bin_num
        self.objective_param.objective = model_meta.objective_meta.objective
        self.objective_param.params = list(model_meta.objective_meta.param)
        self.task_type = model_meta.task_type
        self.n_iter_no_change = model_meta.n_iter_no_change
        self.tol = model_meta.tol

    def set_model_param(self, model_param):
        self.boosting_model_list = list(model_param.trees_)
        self.init_score = np.array(list(model_param.init_score))
        self.history_loss = list(model_param.losses)
        self.classes_ = list(model_param.classes_)
        self.booster_dim = model_param.tree_dim
        self.num_classes = model_param.num_classes
        self.feature_name_fid_mapping.update(model_param.feature_name_fid_mapping)