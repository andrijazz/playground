import torch.nn as nn

activations = dict(
    relu=nn.ReLU(),
    tanh=nn.Tanh()
)


class MLPNet(nn.Module):
    def __init__(self, config):
        super(MLPNet, self).__init__()
        self.config = config

        assert len(self.config.ARCH) >= 2    # mlp conf needs to have at least input dim and output dim
        in_dim = self.config.ARCH[0]
        layers = []
        prev_dim = in_dim
        for i in range(1, len(self.config.ARCH)):
            layers.append(nn.Linear(prev_dim, self.config.ARCH[i], bias=False))
            # avoiding bias for simplicity sake
            if i == len(self.config.ARCH) - 1 and self.config.ADD_OUTPUT_LAYER_ACTIVATION:
                layers.append(activations[self.config.ACTIVATION])
                break
            layers.append(activations[self.config.ACTIVATION])
            prev_dim = self.config.ARCH[i]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        l1_loss = self.l1()
        # print(l1_loss/20)
        return self.net(x) + l1_loss

    def l1(self):
        lmbd = self.config.TRAIN_L1
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

        return lmbd * l1_reg

