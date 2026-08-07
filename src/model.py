import torch.nn as nn


class BreastCancerMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        activation_cls=nn.ReLU,
        norm_type=None,
        dropout_rate=0.0,
        output_dim=1,
        init_method="default",
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if norm_type == "batchnorm":
                layers.append(nn.BatchNorm1d(h_dim))
            elif norm_type == "layernorm":
                layers.append(nn.LayerNorm(h_dim))
            layers.append(activation_cls())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim

        # saida final: logits (sem sigmoid) -- usar com BCEWithLogitsLoss, mais estavel numericamente
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self._init_weights(init_method)

    def _init_weights(self, method):
        if method == "default":
            return
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                if method == "xavier":
                    nn.init.xavier_uniform_(layer.weight)
                elif method == "he":
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.network(x)
