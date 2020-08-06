import torch.nn as nn

activations = dict(
    relu=nn.ReLU(),
    tanh=nn.Tanh()
)


class MLPNet(nn.Module):
    def __init__(self, config):
        super(MLPNet, self).__init__()
        self.config = config
        # mlp conf needs to have at least input dim and output dim
        assert len(self.config.HIDDEN_LAYERS) >= 2
        in_dim = self.config.HIDDEN_LAYERS[0]
        layers = []
        prev_dim = in_dim
        n = len(self.config.HIDDEN_LAYERS)
        for i in range(1, n - 1):
            # avoiding bias for simplicity sake
            layers.append(nn.Linear(prev_dim, self.config.HIDDEN_LAYERS[i], bias=False))
            layers.append(activations[self.config.HIDDEN_ACTIVATION])
            prev_dim = self.config.HIDDEN_LAYERS[i]
        layers.append(nn.Linear(prev_dim, self.config.HIDDEN_LAYERS[n - 1], bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.config.TRAIN_L1 > 0:
            l1_loss = self.l1()
            return self.net(x) + l1_loss
        return self.net(x)

    def l1(self):
        l1_reg = None
        i = 0
        for W in self.parameters():
            # don't apply l1 to last layer
            if i == len(self.config.ARCH) - 1:
                break

            if l1_reg is None:
                l1_reg = W.norm(1)
            else:
                l1_reg = l1_reg + W.norm(1)
            i += 1

        return self.config.TRAIN_L1 * l1_reg

