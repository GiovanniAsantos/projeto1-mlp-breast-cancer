import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=16, output_dim=1, activation="relu", init_method="default"):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = self._get_activation(activation)
        self._init_weights(init_method)

    def _get_activation(self, name):
        activations = {"relu": nn.ReLU(), "tanh": nn.Tanh()}
        return activations[name]

    def _init_weights(self, method):
        if method == "default":
            return
        for layer in (self.fc1, self.fc2):
            if method == "xavier":
                nn.init.xavier_uniform_(layer.weight)
            elif method == "he":
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        # retorna logits (sem sigmoid) -- usar com BCEWithLogitsLoss, mais estavel numericamente
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
