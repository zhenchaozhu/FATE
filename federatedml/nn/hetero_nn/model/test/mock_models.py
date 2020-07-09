import tempfile

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()


def get_data():
    print("[DEBUG] get data and label")
    data = np.array([[0.11, 0.61, 0.22, 0.32],
                     [0.22, 0.39, 0.14, 0.18]])
    label = np.array([[1], [0]])
    return data, label


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
    def __init__(self, input_dim, output_dim, optimizer):
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

        self.optimizer = self._init_optimizer()

    def _init_optimizer(self):
        return torch.optim.SGD(self.parameters(), lr=1.0)

    def forward(self, x):
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

    def backward(self, x, grads):
        msg = "[DEBUG] MockBottomDenseModel.backward"
        LOGGER.debug(msg)
        print(msg)

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

        self.optimizer.step()
        self.optimizer.zero_grad()

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
        pos_prob = torch.sigmoid(x)
        return pos_prob

    def evaluate(self, x, y):
        msg = "[DEBUG] top model start to evaluate"
        LOGGER.debug(msg)
        print(msg)

        x = torch.tensor(x).float()
        y = torch.tensor(y).long()
        y = y.reshape(-1, 1).type_as(x)
        class_loss = self.classifier_criterion(x, y)
        pos_prob = torch.sigmoid(x)
        auc = roc_auc_score(y, pos_prob)

        return {"AUC": auc, "ks": 0.0, "loss": class_loss.item()}

    def export_model(self):
        return ''.encode()

    def restore_model(self, model_bytes):
        pass
