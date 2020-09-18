import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from arch.api.utils import log_utils
from federatedml.nn.hetero_nn.backend.pytorch.interactive.dense_model import InternalDenseModel

LOGGER = log_utils.getLogger()


def get_data():
    print("[DEBUG] get data and label")
    data = np.array([[0.11, 0.61, 0.22, 0.32],
                     [0.22, 0.39, 0.14, 0.18]])
    label = np.array([[1], [0]])
    return data, label


def adjust_learning_rate(lr_0, **kwargs):
    epochs = kwargs["epochs"]
    num_batch = kwargs["num_batch"]
    curr_epoch = kwargs["current_epoch"]
    batch_idx = kwargs["batch_idx"]
    start_steps = curr_epoch * num_batch
    total_steps = epochs * num_batch
    p = float(batch_idx + start_steps) / total_steps

    beta = 0.75
    alpha = 10
    lr = lr_0 / (1 + alpha * p) ** beta
    return lr


class MockInternalDenseModel(InternalDenseModel):
    def __init__(self, input_dim, output_dim):
        super(MockInternalDenseModel, self).__init__(input_dim, output_dim)
        msg = f"[DEBUG] create MockInternalDenseModel with shape [{input_dim}, {output_dim}]"
        LOGGER.debug(msg)
        print(msg)

        if input_dim == 4:
            weight = [[0.51, 0.82, 1.10, 0.3],
                      [0.32, 0.13, 0.13, 0.2],
                      [0.91, 0.22, 1.31, 0.4]]
        elif input_dim == 3:
            weight = [[0.23, 0.19, 1.4]]
        elif input_dim == 2:
            weight = [[0.23, 0.19]]
        else:
            raise Exception(f"Does not support input_shape:{input_dim}")
        bias = [0.01]
        self._set_parameters(weight, bias)


class MockBottomDenseModel(nn.Module):
    def __init__(self, input_dim, output_dim, optimizer_param):
        super(MockBottomDenseModel, self).__init__()
        msg = f"[DEBUG] create host bottom model with shape [{input_dim}, {output_dim}]"
        LOGGER.debug(msg)
        print(msg)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=output_dim),
            # nn.LeakyReLU()
        )

        def init_weights(m):
            if type(m) == nn.Linear:
                with torch.no_grad():
                    if input_dim == 4:
                        m.weight.copy_(torch.tensor([[0.51, 0.82, 1.10, 0.3],
                                                     [0.32, 0.13, 0.13, 0.2],
                                                     [0.91, 0.22, 1.31, 0.4]]))
                    elif input_dim == 3:
                        m.weight.copy_(torch.tensor([[0.23, 0.19, 1.4]]))
                    else:
                        raise Exception("Does not support input_shape:{}")
                m.bias.data.fill_(0.01)

        self.classifier.apply(init_weights)
        self._init_optimizer(optimizer_param)

    def _init_optimizer(self, optimizer_param):
        self.original_learning_rate = optimizer_param.kwargs["learning_rate"]

    def forward(self, x, **kwargs):
        msg = "[DEBUG] MockBottomDenseModel.forward"
        LOGGER.debug(msg)
        print(msg)

        x = torch.tensor(x).float()
        return self.classifier(x).detach().numpy()

    def predict(self, x):
        msg = "[DEBUG] MockBottomDenseModel.predict"
        LOGGER.debug(msg)
        print(msg)

        x = torch.tensor(x).float()
        return self.classifier(x).detach().numpy()

    def backward(self, x, grads, **kwargs):
        msg = "[DEBUG] MockBottomDenseModel.backward"
        LOGGER.debug(msg)
        print(msg)

        msg = f"[DEBUG] host bottom model back-propagation grads:{grads}"
        LOGGER.debug(msg)
        print(msg)

        msg = f"[DEBUG] kwargs:{kwargs}"
        LOGGER.debug(msg)
        print(msg)

        curr_lr = adjust_learning_rate(lr_0=self.original_learning_rate, **kwargs)
        msg = f"[DEBUG] original_learning_rate:{self.original_learning_rate}, adjusted learning rate:{curr_lr}"
        LOGGER.debug(msg)
        print(msg)

        optimizer = optim.SGD(self.parameters(), lr=self.original_learning_rate)
        optimizer.zero_grad()

        x = torch.tensor(x).float()
        grads = torch.tensor(grads).float()
        output = self.classifier(x)
        output.backward(gradient=grads)

        msg = "[DEBUG] Show MockBottomDenseModel params:"
        LOGGER.debug(msg)
        print(msg)

        for name, param in self.named_parameters():
            msg = f"{name}:{param}, grad:\n:{param.grad}"
            LOGGER.debug(msg)
            print(msg)

        optimizer.step()

        msg = "[DEBUG] Show MockBottomDenseModel params after optimized:"
        LOGGER.debug(msg)
        print(msg)
        for name, param in self.named_parameters():
            msg = f"{name}:{param}, grad:\n:{param.grad}"
            LOGGER.debug(msg)
            print(msg)

    def set_data_converter(self, converter):
        pass

    def export_model(self):
        f = tempfile.TemporaryFile()
        try:
            torch.save(self.state_dict(), f)
            f.seek(0)
            model_bytes = f.read()
            return model_bytes
        finally:
            f.close()

    def restore_model(self, model_bytes):
        f = tempfile.TemporaryFile()
        f.write(model_bytes)
        f.seek(0)
        self.load_state_dict(torch.load(f))
        f.close()
