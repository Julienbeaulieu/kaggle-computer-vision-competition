from torch import nn
from yacs.config import CfgNode
from ..layers.linear import LinearLayer
from .build import HEAD_REGISTRY


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

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):

        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        return x