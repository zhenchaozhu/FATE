import functools

from federatedml.ensemble.basic_algorithms import HeteroDecisionTreeGuest
import federatedml.ensemble.boosting.hetero.hetero_fast_secureboost_plan as plan
from federatedml.util import consts

from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()


class HeteroFastDecisionTreeGuest(HeteroDecisionTreeGuest):

    def __init__(self, tree_param):
        super(HeteroFastDecisionTreeGuest, self).__init__(tree_param)
        self.node_plan = []
        self.node_plan_idx = 0
        self.tree_type = consts.MIX_TREE
        self.target_host_id = -1
        self.guest_depth = 0
        self.host_depth = 0
        self.cur_dep = 0
        self.use_guest_feat_when_predict = False

    def use_guest_feat_only_predict_mode(self):
        self.use_guest_feat_when_predict = True

    def set_tree_work_mode(self, tree_type, target_host_id):
        self.tree_type, self.target_host_id = tree_type, target_host_id

    def set_layered_depth(self, guest_depth, host_depth):
        self.guest_depth, self.host_depth = guest_depth, host_depth

    def initialize_node_plan(self):
        if self.tree_type == plan.tree_type_dict['layered_tree']:
            self.node_plan = plan.create_layered_tree_node_plan(guest_depth=self.guest_depth,
                                                                host_depth=self.host_depth,
                                                                host_list=self.host_party_idlist)
            self.max_depth = len(self.node_plan)
            LOGGER.debug('max depth reset to {}, cur node plan is {}'.format(self.max_depth, self.node_plan))
        else:
            self.node_plan = plan.create_node_plan(self.tree_type, self.target_host_id, self.max_depth)

    def get_node_plan(self, idx):
        return self.node_plan[idx]

    def host_id_to_idx(self, host_id):
        if host_id == -1:
            return -1
        return self.host_party_idlist.index(host_id)

    def compute_best_splits_with_node_plan(self, tree_action, target_host_idx, node_map: dict, dep: int,
                                           batch_idx: int):

        LOGGER.debug('node plan at dep {} is {}'.format(dep, (tree_action, target_host_idx)))

        cur_best_split = []

        if tree_action == plan.tree_actions['guest_only']:
            acc_histograms = self.get_local_histograms(node_map)
            cur_best_split = self.splitter.find_split(acc_histograms, self.valid_features,
                                                      self.data_bin._partitions, self.sitename,
                                                      self.use_missing, self.zero_as_missing)
            LOGGER.debug('computing local splits done')

        if tree_action == plan.tree_actions['host_only']:

            self.federated_find_split(dep, batch_idx, idx=target_host_idx)
            host_split_info = self.sync_final_split_host(dep, batch_idx, idx=target_host_idx)

            LOGGER.debug('get encrypted split value from host')

            cur_best_split = self.merge_splitinfo(splitinfo_guest=[],
                                                  splitinfo_host=host_split_info,
                                                  merge_host_split_only=True)

        return cur_best_split

    def assign_instances_to_new_node_with_node_plan(self, dep, tree_action, target_host_idx):

        LOGGER.info("redispatch node of depth {}".format(dep))
        dispatch_node_method = functools.partial(self.assign_a_instance,
                                                 tree_=self.tree_node,
                                                 decoder=self.decode,
                                                 sitename=self.sitename,
                                                 split_maskdict=self.split_maskdict,
                                                 bin_sparse_points=self.bin_sparse_points,
                                                 use_missing=self.use_missing,
                                                 zero_as_missing=self.zero_as_missing,
                                                 missing_dir_maskdict=self.missing_dir_maskdict)

        dispatch_guest_result = self.data_with_node_assignments.mapValues(dispatch_node_method)
        LOGGER.info("remask dispatch node result of depth {}".format(dep))

        dispatch_to_host_result = dispatch_guest_result.filter(
            lambda key, value: isinstance(value, tuple) and len(value) > 2)

        dispatch_guest_result = dispatch_guest_result.subtractByKey(dispatch_to_host_result)
        leaf = dispatch_guest_result.filter(lambda key, value: isinstance(value, tuple) is False)
        if self.sample_weights is None:
            self.sample_weights = leaf
        else:
            self.sample_weights = self.sample_weights.union(leaf)

        dispatch_guest_result = dispatch_guest_result.subtractByKey(leaf)

        if tree_action == plan.tree_actions['host_only']:
            dispatch_node_host_result = self.sync_dispatch_node_host(dispatch_to_host_result, dep, idx=
                                                                     target_host_idx)

            self.inst2node_idx = None
            for idx in range(len(dispatch_node_host_result)):
                if self.inst2node_idx is None:
                    self.inst2node_idx = dispatch_node_host_result[idx]
                else:
                    self.inst2node_idx = self.inst2node_idx.join(dispatch_node_host_result[idx],
                                                                 lambda unleaf_state_nodeid1, unleaf_state_nodeid2:
                                                                 unleaf_state_nodeid1 if len(
                                                                     unleaf_state_nodeid1) == 2 else unleaf_state_nodeid2)
            self.inst2node_idx = self.inst2node_idx.union(dispatch_guest_result)

        else:
            self.inst2node_idx = dispatch_guest_result

    def fit(self):

        LOGGER.debug('fitting a hetero decision tree')

        self.initialize_node_plan()

        if self.tree_type == plan.tree_type_dict['host_feat_only'] or \
                self.tree_type == plan.tree_type_dict['layered_tree']:
            self.sync_encrypted_grad_and_hess(idx=self.host_id_to_idx(self.target_host_id))

        root_node = self.initialize_root_node()
        self.cur_layer_nodes = [root_node]
        self.inst2node_idx = self.assign_instance_to_root_node(self.data_bin, root_node_id=root_node.id)

        for dep in range(self.max_depth):

            tree_action, layer_target_host_id = self.get_node_plan(dep)
            host_idx = self.host_id_to_idx(layer_target_host_id)

            self.sync_cur_to_split_nodes(self.cur_layer_nodes, dep)
            if len(self.cur_layer_nodes) == 0:
                break

            self.sync_node_positions(dep, idx=host_idx)
            self.data_with_node_assignments = self.data_bin.join(self.inst2node_idx, lambda data_inst, dispatch_info: (
                data_inst, dispatch_info))

            split_info = []
            for batch_idx, i in enumerate(range(0, len(self.cur_layer_nodes), self.max_split_nodes)):
                self.cur_to_split_nodes = self.cur_layer_nodes[i: i + self.max_split_nodes]
                cur_splitinfos = self.compute_best_splits_with_node_plan(tree_action, host_idx, node_map=
                                                                         self.get_node_map(self.cur_to_split_nodes),
                                                                         dep=dep, batch_idx=batch_idx)
                split_info.extend(cur_splitinfos)

            reach_max_depth = True if dep + 1 == self.max_depth else False

            self.update_tree(split_info, reach_max_depth)
            self.assign_instances_to_new_node_with_node_plan(dep, tree_action, host_idx)

        self.convert_bin_to_real()
        self.sync_tree()
        LOGGER.info("tree node num is %d" % len(self.tree_node))
        LOGGER.info("end to fit guest decision tree")


    def predict(self, data_inst):

        LOGGER.info("start to predict!")
        predict_data = data_inst.mapValues(lambda inst: (0, 1))
        site_host_send_times = 0
        predict_result = None

        while True:
            traverse_tree = functools.partial(self.traverse_tree,
                                              tree_=self.tree_node,
                                              decoder=self.decode,
                                              sitename=self.sitename,
                                              split_maskdict=self.split_maskdict,
                                              use_missing=self.use_missing,
                                              zero_as_missing=self.zero_as_missing,
                                              missing_dir_maskdict=self.missing_dir_maskdict)
            predict_data = predict_data.join(data_inst, traverse_tree)
            predict_leaf = predict_data.filter(lambda key, value: isinstance(value, tuple) is False)

            if predict_result is None:
                predict_result = predict_leaf
            else:
                predict_result = predict_result.union(predict_leaf)

            predict_data = predict_data.subtractByKey(predict_leaf)
            LOGGER.debug('showing predict data {}'.format(predict_data))

            unleaf_node_count = predict_data.count()

            if self.use_guest_feat_when_predict:
                break

            if unleaf_node_count == 0:
                self.sync_predict_finish_tag(True, site_host_send_times)
                break

            self.sync_predict_finish_tag(False, site_host_send_times)
            self.sync_predict_data(predict_data, site_host_send_times)

            predict_data_host = self.sync_data_predicted_by_host(site_host_send_times)
            for i in range(len(predict_data_host)):
                predict_data = predict_data.join(predict_data_host[i],
                                                 lambda state1_nodeid1, state2_nodeid2:
                                                 state1_nodeid1 if state1_nodeid1[
                                                                       1] == 0 else state2_nodeid2)

            site_host_send_times += 1

        LOGGER.info("predict finish!")
        return predict_result

