import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.autograd import Function

from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()

activation_fn = nn.LeakyReLU()


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class RegionalFeatureExtractor(nn.Module):
    def __init__(self, input_dims):
        super(RegionalFeatureExtractor, self).__init__()
        print("input_dims:", input_dims)
        if len(input_dims) == 4:
            self.extractor = nn.Sequential(
                nn.Linear(in_features=input_dims[0], out_features=input_dims[1]),
                nn.BatchNorm1d(input_dims[1]),
                activation_fn,
                nn.Linear(in_features=input_dims[1], out_features=input_dims[2]),
                nn.BatchNorm1d(input_dims[2]),
                activation_fn,
                nn.Linear(in_features=input_dims[2], out_features=input_dims[3]),
                nn.BatchNorm1d(input_dims[3]),
                activation_fn
            )
        elif len(input_dims) == 3:
            self.extractor = nn.Sequential(
                nn.Linear(in_features=input_dims[0], out_features=input_dims[1]),
                nn.BatchNorm1d(input_dims[1]),
                activation_fn,
                nn.Linear(in_features=input_dims[1], out_features=input_dims[2]),
                nn.BatchNorm1d(input_dims[2]),
                activation_fn
            )
        elif len(input_dims) == 2:
            self.extractor = nn.Sequential(
                nn.Linear(in_features=input_dims[0], out_features=input_dims[1]),
                nn.BatchNorm1d(input_dims[1]),
                activation_fn
            )
        else:
            raise RuntimeError(f"Currently does not support input_dims of layers {input_dims}")

    def forward(self, x):
        x = self.extractor(x)
        return x


class RegionalAggregator(nn.Module):
    def __init__(self, input_dim):
        super(RegionalAggregator, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=1),
            activation_fn
        )

    def forward(self, x):
        return self.classifier(x)


class RegionalDiscriminator(nn.Module):

    def __init__(self, input_dim):
        super(RegionalDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=30),
            nn.BatchNorm1d(30),
            activation_fn,
            nn.Linear(in_features=30, out_features=10),
            nn.BatchNorm1d(10),
            activation_fn,
            nn.Linear(in_features=10, out_features=2)
        )

    def apply_discriminator(self, x):
        return self.discriminator(x)

    def forward(self, input, alpha):
        reversed_input = ReverseLayerF.apply(input, alpha)
        x = self.apply_discriminator(reversed_input)
        return x


# following code is for creating guest top model

class TopModel(object):
    def __init__(self):
        self.classifier_criterion = nn.BCEWithLogitsLoss(reduction='mean')

    def set_data_converter(self, data_converter):
        pass

    def train_and_get_backward_gradient(self, x, y):
        LOGGER.debug("[DEBUG] MockTopModel.train_and_get_backward_gradient:")

        x = torch.tensor(x, requires_grad=True).float()
        y = torch.tensor(y).long()

        pos_prob = torch.sigmoid(x.flatten())
        auc = roc_auc_score(y.tolist(), pos_prob.tolist())

        y = y.reshape(-1, 1).type_as(x)
        class_loss = self.classifier_criterion(x, y)
        grads = torch.autograd.grad(outputs=class_loss, inputs=x)

        LOGGER.debug(f"[DEBUG] class_loss={class_loss}, auc:{auc}")
        LOGGER.debug(f"[DEBUG] top model back-propagation grads={grads}")

        return grads[0].numpy()

    def predict(self, x):
        LOGGER.debug("[DEBUG] top model start to predict")

        x = torch.tensor(x).float()
        pred_prob = torch.sigmoid(x.flatten())
        return pred_prob.numpy().reshape(-1, 1)

    def evaluate(self, x, y):
        LOGGER.debug("[DEBUG] top model start to evaluate")

        x = torch.tensor(x).float()
        y = torch.tensor(y).long()

        pred_prob = torch.sigmoid(x.flatten())
        pred_y = torch.round(pred_prob).long()
        y = y.reshape(-1, 1).type_as(x)

        auc = roc_auc_score(y.tolist(), pred_prob.tolist())
        acc = accuracy_score(y.tolist(), pred_y.tolist())
        class_loss = self.classifier_criterion(x, y)
        return {"loss": class_loss.item(), "auc": auc, "acc": acc}

    def export_model(self):
        return ''.encode()

    def restore_model(self, model_bytes):
        pass
