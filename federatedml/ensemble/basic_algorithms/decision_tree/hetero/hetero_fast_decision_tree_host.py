from federatedml.ensemble.basic_algorithms import HeteroDecisionTreeHost

import federatedml.ensemble.boosting.hetero.hetero_fast_secureboost_plan as plan
from federatedml.util import consts

from arch.api.utils import log_utils
import functools

LOGGER = log_utils.getLogger()


class HeteroFastDecisionTreeHost(HeteroDecisionTreeHost):

    def __init__(self, tree_param):
        super(HeteroFastDecisionTreeHost, self).__init__(tree_param)
        self.node_plan = []
        self.node_plan_idx = 0
        self.tree_type = consts.MIX_TREE
        self.target_host_id = -1
        self.guest_depth = 0
        self.host_depth = 0
        self.cur_dep = 0
        self.self_host_id = -1
        self.use_guest_feat_when_predict = False

    def use_guest_feat_only_predict_mode(self):
        self.use_guest_feat_when_predict = True

    def set_tree_work_mode(self, tree_type, target_host_id):
        self.tree_type, self.target_host_id = tree_type, target_host_id

    def set_layered_depth(self, guest_depth, host_depth):
        self.guest_depth, self.host_depth = guest_depth, host_depth

    def set_self_host_id(self, self_host_id):
        self.self_host_id = self_host_id

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

    def compute_best_splits_with_node_plan(self, tree_action, target_host_id, node_map: dict, dep: int, batch_idx: int):

        LOGGER.debug('node plan at dep {} is {}'.format(dep, (tree_action, target_host_id)))

        if tree_action == plan.tree_actions['host_only'] and target_host_id == self.self_host_id:
            acc_histograms = self.get_local_histograms(node_map)
            splitinfo_host, encrypted_splitinfo_host = self.splitter.find_split_host(acc_histograms,
                                                                                     self.valid_features,
                                                                                     self.data_bin._partitions,
                                                                                     self.sitename,
                                                                                     self.use_missing,
                                                                                     self.zero_as_missing)

            self.sync_encrypted_splitinfo_host(encrypted_splitinfo_host, dep, batch_idx)
            federated_best_splitinfo_host = self.sync_federated_best_splitinfo_host(dep, batch_idx)
            self.sync_final_splitinfo_host(splitinfo_host, federated_best_splitinfo_host, dep, batch_idx)
            LOGGER.debug('computing host splits done')

        else:
            LOGGER.debug('skip best split computation')

    def fit(self):

        LOGGER.info("begin to fit fast host decision tree")

        self.initialize_node_plan()

        if self.tree_type == plan.tree_type_dict['guest_feat_only']:
            LOGGER.debug('this tree uses guest feature only, skip')

        elif self.self_host_id == self.target_host_id or self.tree_type == plan.tree_type_dict['layered_tree']:

            LOGGER.debug('use host feature to build tree')
            self.sync_encrypted_grad_and_hess()

            for dep in range(self.max_depth):

                tree_action, layer_target_host_id = self.get_node_plan(dep)

                self.sync_tree_node_queue(dep)
                if len(self.cur_layer_nodes) == 0:
                    break

                if self.self_host_id == layer_target_host_id:
                    self.inst2node_idx = self.sync_node_positions(dep)
                    self.data_with_node_assignments = self.data_bin.join(self.inst2node_idx, lambda v1, v2: (v1, v2))

                batch = 0
                for i in range(0, len(self.cur_layer_nodes), self.max_split_nodes):
                    self.cur_to_split_nodes = self.cur_layer_nodes[i: i + self.max_split_nodes]
                    self.compute_best_splits_with_node_plan(tree_action, layer_target_host_id,
                                                            node_map=self.get_node_map(self.cur_to_split_nodes),
                                                            dep=dep, batch_idx=batch, )
                    batch += 1

                if layer_target_host_id == self.self_host_id:
                    dispatch_node_host = self.sync_dispatch_node_host(dep)
                    self.assign_instances_to_new_node(dispatch_node_host, dep)

        self.sync_tree()
        self.convert_bin_to_real()

        LOGGER.info("end to fit guest decision tree")

    def predict(self, data_inst):
        LOGGER.info("start to predict!")
        site_guest_send_times = 0

        if self.use_guest_feat_when_predict:
            LOGGER.debug('use guest feat tree to predict, skip')
            return

        while True:
            finish_tag = self.sync_predict_finish_tag(site_guest_send_times)
            if finish_tag is True:
                break

            predict_data = self.sync_predict_data(site_guest_send_times)

            traverse_tree = functools.partial(self.traverse_tree,
                                              tree_=self.tree_node,
                                              decoder=self.decode,
                                              split_maskdict=self.split_maskdict,
                                              sitename=self.sitename,
                                              use_missing=self.use_missing,
                                              zero_as_missing=self.zero_as_missing,
                                              missing_dir_maskdict=self.missing_dir_maskdict)
            predict_data = predict_data.join(data_inst, traverse_tree)

            self.sync_data_predicted_by_host(predict_data, site_guest_send_times)

            site_guest_send_times += 1

        LOGGER.info("predict finish!")