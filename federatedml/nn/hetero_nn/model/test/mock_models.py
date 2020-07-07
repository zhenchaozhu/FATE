import tempfile

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def get_data():
    print("get data and label")
    data = np.array([[0.11, 0.61, 0.22, 0.32],
                     [0.22, 0.39, 0.14, 0.18]])
    label = np.array([[1], [0]])
    return data, label


class MockDenseModel(nn.Module):
    def __init__(self, input_dim, output_dim, optimizer=None):
        super(MockDenseModel, self).__init__()
        print(f"[DEBUG] create dense layer with shape [{input_dim}, {output_dim}]")
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
        print(f"[DEBUG] create Bottom model with shape [{input_dim}, {output_dim}]")
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
        print("[DEBUG] MockBottomDenseModel.forward")
        x = torch.tensor(x).float()
        return self.classifier(x).detach().numpy()

    def backward(self, x, grads):
        print("[DEBUG] MockBottomDenseModel.backward")
        x = torch.tensor(x).float()
        grads = torch.tensor(grads).float()
        output = self.classifier(x)
        output.backward(gradient=grads)

        print("[DEBUG] Show MockBottomDenseModel params:")
        for name, param in self.named_parameters():
            print(f"{name}:{param}, grad:\n:{param.grad}")

        self.optimizer.step()
        self.optimizer.zero_grad()

        print("[DEBUG] Show MockBottomDenseModel params after optimized:")
        for name, param in self.named_parameters():
            print(f"{name}:{param}, grad:\n:{param.grad}")

    def set_data_converter(self, converter):
        pass


class MockTopModel(object):
    def __init__(self, input_shape=None, loss=None, optimizer="SGD", metrics=None, model_builder=None,
                 layer_config=None):
        # self._model = model_builder(input_shape=input_shape,
        #                             nn_define=layer_config,
        #                             optimizer=optimizer,
        #                             loss=loss,
        #                             metrics=metrics)
        #
        # self.data_converter = None
        self.classifier_criterion = nn.BCEWithLogitsLoss(reduction='mean')

    def set_data_converter(self, data_converter):
        pass

    def train_and_get_backward_gradient(self, x, y):
        print("[DEBUG] MockTopModel.train_and_get_backward_gradient")
        x = torch.tensor(x, requires_grad=True).float()
        print("x=", x)
        y = torch.tensor(y).long()
        print("y=", y)
        y = y.reshape(-1, 1).type_as(x)

        class_loss = self.classifier_criterion(x, y)
        print("[DEBUG] *class_loss=", class_loss)
        grads = torch.autograd.grad(outputs=class_loss, inputs=x)
        print("[DEBUG] *top model back-propagation grads=", grads)
        return grads[0].numpy()

    def predict(self, x):
        print("[DEBUG] top model start to predict")
        x = torch.tensor(x).float()
        pos_prob = torch.sigmoid(x)
        return pos_prob

    def evaluate(self, x, y):
        print("[DEBUG] top model start to evaluate")
        x = torch.tensor(x).float()
        y = torch.tensor(y).long()
        y = y.reshape(-1, 1).type_as(x)
        class_loss = self.classifier_criterion(x, y)
        pos_prob = torch.sigmoid(x)
        auc = roc_auc_score(y, pos_prob)

        return {"AUC": auc, "ks": 0.0, "loss": class_loss.item()}

    def export_model(self):
        return None

    def restore_model(self, model_bytes):
        pass
