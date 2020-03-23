from torch import nn
from yacs.config import CfgNode
from ..layers.linear import LinearLayer
from .build import HEAD_REGISTRY

# The Registy.register decorator must be invoked as a function, with the desired paramters (here 'simple_head'), like so:
@HEAD_REGISTRY.register('simple_head')
def build_simple_pred_head(head_cfg: CfgNode) -> nn.Module:
    return SimplePredictionHead(head_cfg)


class SimplePredictionHead(nn.Module):

    def __init__(self, head_cfg: CfgNode):
        super(SimplePredictionHead, self).__init__()
        self.fc_layers = []
        input_dim = head_cfg.INPUT_DIM
        # first hidden layers
        for hidden_dim in head_cfg.HIDDEN_DIMS:
            self.fc_layers.append(
                LinearLayer(input_dim, hidden_dim, bn=head_cfg.BN, activation=head_cfg.ACTIVATION,
                            dropout_rate=head_cfg.DROPOUT)
            )
            input_dim = hidden_dim

        output_dims = head_cfg.OUTPUT_DIMS

        # prediction layer
        self.fc_layers.append(
            LinearLayer(input_dim, sum(output_dims), bn=False, activation=None, dropout_rate=-1)
        )

        self.fc_layers = nn.Sequential(*self.fc_layers)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):

        return self.fc_layers(x)

@HEAD_REGISTRY.register('simple_head_kaiming')
def build_simple_pred_head_kaiming(head_cfg: CfgNode) -> nn.Module:
    return SimplePredictionHeadKaiming(head_cfg)


class SimplePredictionHeadKaiming(nn.Module):

    def __init__(self, head_cfg: CfgNode):
        super(SimplePredictionHeadKaiming, self).__init__()
        self.fc_layers = []
        input_dim = head_cfg.INPUT_DIM
        # first hidden layers
        for hidden_dim in head_cfg.HIDDEN_DIMS:
            self.fc_layers.append(
                LinearLayer(input_dim, hidden_dim, bn=head_cfg.BN, activation=head_cfg.ACTIVATION,
                            dropout_rate=head_cfg.DROPOUT)
            )
            input_dim = hidden_dim

        output_dims = head_cfg.OUTPUT_DIMS

        # prediction layer
        self.fc_layers.append(
            LinearLayer(input_dim, sum(output_dims), bn=False, activation=None, dropout_rate=-1)
        )

        self.fc_layers = nn.Sequential(*self.fc_layers)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):

        return self.fc_layers(x)