import numpy as np
import torch.nn as nn
import torch
from federatedml.nn.hetero_nn.backend.hetero_nn_model import HeteroNNKerasHostModel, HeteroNNKerasGuestModel
from federatedml.param.hetero_nn_param import HeteroNNParam
from federatedml.nn.hetero_nn.model.test.mock_models import get_data


def test_HeteroNNKerasHostModel():
    data, label = get_data()
    hetero_nn_param = HeteroNNParam()
    host_model = HeteroNNKerasHostModel(hetero_nn_param)
    host_model.train(data, 1, 1)


def test_HeteroNNKerasGuestModel():
    data, label = get_data()
    hetero_nn_param = HeteroNNParam()
    guest_model = HeteroNNKerasGuestModel(hetero_nn_param)
    guest_model.set_empty()
    guest_model.train(data, label, 1, 1)


if __name__ == "__main__":
    test_HeteroNNKerasHostModel()
    # test_HeteroNNKerasGuestModel()
