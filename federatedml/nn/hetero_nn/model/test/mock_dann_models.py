from arch.api.utils import log_utils
import torch.nn as nn
import torch

LOGGER = log_utils.getLogger()


class MockRegionalExtractorModel(nn.Module):
    def __init__(self, input_dim=4, output_dim=2):
        super(MockRegionalExtractorModel, self).__init__()
        msg = f"[DEBUG] create MockRegionalExtractorModel with shape [{input_dim}, {output_dim}]"
        LOGGER.debug(msg)
        print(msg)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=output_dim),
            # nn.LeakyReLU()
        )

        weight = [[0.51, 0.82, 1.10, 0.3],
                  [0.32, 0.13, 0.13, 0.2]]
        bias = 0.01
        self._set_parameters(weight, bias)

    def _set_parameters(self, weight, bias):
        def init_weights(m):
            if type(m) == nn.Linear:
                with torch.no_grad():
                    m.weight.copy_(torch.tensor(weight))
                    # m.bias.copy_(torch.tensor(bias))
                    m.bias.data.fill_(torch.tensor(bias))
        self.classifier.apply(init_weights)

    def forward(self, x, **kwargs):
        x = torch.tensor(x).float()
        return self.classifier(x)


class MockRegionalDiscriminatorModel(nn.Module):
    def __init__(self, input_dim=2, output_dim=2):
        super(MockRegionalDiscriminatorModel, self).__init__()
        msg = f"[DEBUG] create MockRegionalDiscriminatorModel with shape [{input_dim}, {output_dim}]"
        LOGGER.debug(msg)
        print(msg)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=output_dim),
            # nn.LeakyReLU()
        )

        weight = [[0.23, 0.19],
                  [0.11, 0.33]]
        bias = 0.01
        self._set_parameters(weight, bias)

    def _set_parameters(self, weight, bias):
        def init_weights(m):
            if type(m) == nn.Linear:
                with torch.no_grad():
                    m.weight.copy_(torch.tensor(weight))
                    # m.bias.copy_(torch.tensor(bias))
                    m.bias.data.fill_(torch.tensor(bias))
        self.classifier.apply(init_weights)

    def forward(self, x, alpha):
        x = torch.tensor(x).float()
        return self.classifier(x)


class MockRegionalAggregatorModel(nn.Module):
    def __init__(self, input_dim=2, output_dim=1):
        super(MockRegionalAggregatorModel, self).__init__()
        msg = f"[DEBUG] create MockRegionalAggregatorModel with shape [{input_dim}, {output_dim}]"
        LOGGER.debug(msg)
        print(msg)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=output_dim),
            nn.LeakyReLU()
        )

        weight = [[0.23, 0.19]]
        bias = 0.01
        self._set_parameters(weight, bias)

    def _set_parameters(self, weight, bias):
        def init_weights(m):
            if type(m) == nn.Linear:
                with torch.no_grad():
                    m.weight.copy_(torch.tensor(weight))
                    # m.bias.copy_(torch.tensor(bias))
                    m.bias.data.fill_(torch.tensor(bias))
        self.classifier.apply(init_weights)

    def forward(self, x, **kwargs):
        x = torch.tensor(x).float()
        return self.classifier(x)
