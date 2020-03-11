from arch.api.utils import log_utils
from federatedml.util import consts
from typing import Tuple, List

LOGGER = log_utils.getLogger()

tree_type_list = {
    'guest_feat_only': 0,  # use only guest feature to build this tree
    'host_feat_only': 1,  # use only host feature to build this tree
    'normal_tree': 2,  # a normal decision tree
    'layered_tree': 3  # a layered decision tree
}

tree_actions = {
    'guest_only': 0,  # use only guest feature to build this layer
    'host_only': 1,  # use only host feature to build this layer
    'guest_and_host': 2,  # use global feature to build this layer
}


def create_tree_plan(work_mode: str, k=1, tree_num=10, host_list=None):

    """
    Args:
        work_mode:
        k: k is needed when work_mode is 'layered'
        tree_num: decision tree number
        host_list: need to specify host idx when under multi-host scenario, default is None
    Returns: tree plan
    """

    LOGGER.info('boosting_core trees work mode is {}'.format(work_mode))
    tree_plan = []
    if work_mode == consts.NORMAL_TREE:
        tree_plan = [(tree_type_list['normal_tree'], -1) for i in range(tree_num)]

    elif work_mode == consts.COMPLETE_SECURE_TREE:
        tree_plan = [(tree_type_list['normal_tree'], -1) for i in range(tree_num)]
        tree_plan.insert(0, (tree_type_list['guest_feat_only'], -1))

    elif work_mode == consts.MIX_TREE:
        assert k > 0
        assert len(host_list) > 0

        one_round = [(tree_type_list['guest_feat_only'], -1)] * k
        for host_idx, host_id in enumerate(host_list):
            one_round += [(tree_type_list['host_feat_only'], host_id)] * k

        round_num = (tree_num // (2*k)) + 1
        tree_plan = (one_round * round_num)[0:tree_num]

    elif work_mode == consts.LAYERED_TREE:
        tree_plan = [(tree_type_list['layered_tree'], -1) for i in range(tree_num)]

    LOGGER.debug('tree plan is {}'.format(tree_plan))
    return tree_plan


def create_node_plan(tree_work_mode_tuple: Tuple[int, int], max_depth,
                     guest_depth=0, host_depth=0, host_list=None) -> List[Tuple[int, int]]:

    """
    Args:
        tree_work_mode_tuple: [tree_type, target_host_id]
        max_depth: tree max depth
        guest_depth: needed when work mode is 3
        host_depth: needed when work mode is 3
        host_list: need to specify host idx when under multi-host scenario, default is None
    Returns:
    """

    LOGGER.debug('cur tree working mode is {}'.format(tree_work_mode_tuple))
    node_plan = []
    tree_type = tree_work_mode_tuple[0]

    if tree_type == tree_type_list['normal_tree']:
        target_host_id = tree_work_mode_tuple[1]
        node_plan = [(tree_actions['guest_and_host'], target_host_id) for i in range(max_depth)]

    elif tree_type == tree_type_list['guest_feat_only']:
        target_host_id = tree_work_mode_tuple[1]
        node_plan = [(tree_actions['guest_only'], target_host_id) for i in range(max_depth)]

    elif tree_type == tree_type_list['host_feat_only']:
        target_host_id = tree_work_mode_tuple[1]
        node_plan = [(tree_actions['host_only'], target_host_id) for i in range(max_depth)]

    elif tree_type == tree_type_list['layered_tree']:
        # max_depth = guest_depth + host_depth*len(host_list)
        node_plan = [tree_actions['host_only']] * host_depth + [tree_actions['guest_only']] * guest_depth

    return node_plan


def encode_plan(p, split_token='-'):

    result = []
    for tree_type_or_action, host_id in p:
        result.append(str(tree_type_or_action)+split_token+str(host_id))
    return result


def decode_plan(s, split_token='-'):

    result = []
    for string in s:
        t = string.split(split_token)
        result.append((t[0], t[1]))

    return result
