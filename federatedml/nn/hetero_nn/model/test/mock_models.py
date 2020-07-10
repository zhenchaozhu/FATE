import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from arch.api.utils import log_utils

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


class MockInternalDenseModel(nn.Module):
    def __init__(self, input_dim, output_dim, optimizer=None):
        super(MockInternalDenseModel, self).__init__()
        msg = f"[DEBUG] create dense layer with shape [{input_dim}, {output_dim}]"
        LOGGER.debug(msg)
        print(msg)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=output_dim),
            # nn.LeakyReLU()
        )

        if input_dim == 4:
            weight = [[0.51, 0.82, 1.10, 0.3],
                      [0.32, 0.13, 0.13, 0.2],
                      [0.91, 0.22, 1.31, 0.4]]
        elif input_dim == 3:
            weight = [[0.23, 0.19, 1.4]]
        else:
            raise Exception("Does not support input_shape:{}")
        bias = [0.01]
        self._set_parameters(weight, bias)

    def _set_parameters(self, weight, bias):
        def init_weights(m):
            if type(m) == nn.Linear:
                with torch.no_grad():
                    m.weight.copy_(torch.tensor(weight))
                    # m.bias.data.fill_(torch.tensor(bias))
                    m.bias.copy_(torch.tensor(bias))

        self.classifier.apply(init_weights)

    def set_parameters(self, parameters):
        weight = parameters["weight"]
        bias = parameters["bias"]
        self._set_parameters(weight, bias)

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
        # self.optimizer = optim.SGD(self.parameters(), lr=self.original_learning_rate, momentum=0.9)

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

        curr_lr = adjust_learning_rate(lr_0=self.original_learning_rate, **kwargs)
        msg = f"[DEBUG] kwargs:{kwargs}"
        LOGGER.debug(msg)
        print(msg)

        msg = f"[DEBUG] adjusted learning rate:{curr_lr}"
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


class MockTopModel(object):
    def __init__(self):
        self.classifier_criterion = nn.BCEWithLogitsLoss(reduction='mean')

    def set_data_converter(self, data_converter):
        pass

    def train_and_get_backward_gradient(self, x, y):
        msg = "[DEBUG] MockTopModel.train_and_get_backward_gradient:"
        LOGGER.debug(msg)
        print(msg)

        x = torch.tensor(x, requires_grad=True).float()
        y = torch.tensor(y).long()
        y = y.reshape(-1, 1).type_as(x)
        print("x=", x)
        print("y=", y)

        class_loss = self.classifier_criterion(x, y)
        grads = torch.autograd.grad(outputs=class_loss, inputs=x)
        msg = f"[DEBUG] *class_loss={class_loss}"
        LOGGER.debug(msg)
        print(msg)
        msg = f"[DEBUG] *top model back-propagation grads={grads}"
        LOGGER.debug(msg)
        print(msg)

        return grads[0].numpy()

    def predict(self, x):
        msg = "[DEBUG] top model start to predict"
        LOGGER.debug(msg)
        print(msg)

        x = torch.tensor(x).float()
        pos_prob = torch.sigmoid(x.flatten())
        return pos_prob.numpy().reshape(-1, 1)

    def evaluate(self, x, y):
        msg = "[DEBUG] top model start to evaluate"
        LOGGER.debug(msg)
        print(msg)

        x = torch.tensor(x).float()
        y = torch.tensor(y).long()
        y = y.reshape(-1, 1).type_as(x)
        class_loss = self.classifier_criterion(x, y)
        return {"loss": class_loss.item()}

    def export_model(self):
        return ''.encode()

    def restore_model(self, model_bytes):
        pass
