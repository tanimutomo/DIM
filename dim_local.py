import torch
import torch.nn as nn
from collections import OrderedDict


class Encoder(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.encoder = Convnet(classifier)

    def forward(self, x):
        x = self.encoder(x)
        return x


class Convnet(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        
        assert classifier in ['conv', 'fc', 'glob']
        self.classifier = classifier

        self.layers = nn.Sequential(OrderedDict([
                            ('layer0', self.make_conv_layer(3, 64)),
                            ('layer1', self.make_conv_layer(64, 128)),
                            ('layer2', self.make_conv_layer(128, 256)),
                            ('layer3', View(-1, 4096)),
                            ('layer4', self.make_fc_layer(4096, 1024)),
                            ('layer5', nn.Linear(1024, 64, bias=True))
                            ]))

    def make_conv_layer(self, in_c, out_c, kernel_size=(4, 4), stride=(2, 2),
                        padding=(1, 1), bias=False, eps=1e-05, momentum=0.1,
                        affine=True, track_running_stats=True, inplace=True):
        conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size,
                         stride=stride, padding=padding, bias=bias)
        bn = nn.BatchNorm2d(out_c, eps=eps, momentum=momentum,
                            affine=affine, 
                            track_running_stats=track_running_stats)
        relu = nn.ReLU(inplace=inplace)
        return nn.Sequential(OrderedDict([('conv', conv),
                                          ('bn', bn),
                                          ('ReLU', relu)]))

    def make_fc_layer(self, in_f, out_f, bias=False, eps=1e-05, momentum=0.1,
                      affine=True, track_running_stats=True, inplace=True):
        fc = nn.Linear(in_f, out_f, bias=bias)
        bn = nn.BatchNorm1d(out_f, eps=eps, momentum=momentum,
                            affine=affine,
                            track_running_stats=track_running_stats)
        relu = nn.ReLU(inplace=inplace)
        return nn.Sequential(OrderedDict([('fc', fc),
                                          ('bn', bn),
                                          ('ReLU', relu)]))

    def forward(self, x):
        x = self.layers.layer0(x)
        x = self.layers.layer1(x)
        x = self.layers.layer2(x)
        if self.classifier == 'conv':
            return x
        x = self.layers.layer3(x)
        if self.classifier == 'fc':
            return x
        x = self.layers.layer4(x)
        return x


class View(torch.nn.Module):
    """Basic reshape module.
    """
    def __init__(self, *shape):
        """
        Args:
            *shape: Input shape.
        """
        super().__init__()
        self.shape = shape

    def forward(self, input):
        """Reshapes tensor.
        Args:
            input: Input tensor.
        Returns:
            torch.Tensor: Flattened tensor.
        """
        return input.view(*self.shape)


if __name__ == '__main__':
    classifier = 'conv'
    encoder = Encoder(classifier)
    encoder.load_state_dict(torch.load('./models/dim_local_dv_encoder.pth'))
    print(encoder)
    # encoder = encoder.encoder
    # print(encoder)

    # import torch.optim as optim
    # classifier = 'conv'
    # encoder = BigEncoder(classifier)
    # # encoder.load_state_dict(torch.load('./models/dim_local_dv_encoder.pth'))
    # optimizer = optim.Adam(encoder.parameters())
    # optimizer.zero_grad()
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    out = encoder(x)
    print(out.shape)
    # criterion = nn.L1Loss()
    # t = torch.ones(out.shape)
    # loss = criterion(out, t)
    # loss.backward()
    # optimizer.step()
    # torch.save(encoder.state_dict(), 'models/tmp.pth')


