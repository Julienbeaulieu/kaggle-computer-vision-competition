from torch import nn
import torch.nn.functional as F
from typing import Union

ACTIVATION_FN = {
    'relu': F.relu,
    'relu6': F.relu6,
    'elu': F.elu,
    'leaky_relu': F.leaky_relu,
    None: None
}


class LinearLayer(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, activation: Union[None, str], bn: bool,
                 dropout_rate: float = -1):
        super(LinearLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation_fn = ACTIVATION_FN[activation]
        if bn:
            self.bn = nn.BatchNorm1d(self.output_dim)
        else:
            self.bn = None
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        # LINEAR -> BN -> ACTIVATION -> DROPOUT
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x, inplace=True)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
